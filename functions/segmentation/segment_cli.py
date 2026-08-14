"""Command line entry point that the MATLAB Retrospective app calls.

This module is the bridge between the app and the segmentation model: it reads
the .mat file MATLAB writes, runs the model over it with onnxruntime, and writes
the label maps back as a .mat file, in the same style the DRIM reconstruction
already uses.

    python segment_cli.py <data folder> <output folder> [clean|raw] [max_dist_px]

Files exchanged, all inside those folders:

    retroLVtemp.mat       in   images + geometry, written by MATLAB
    retroLVtemp_SEG.mat   out  label maps, read back by MATLAB
    retroLVseg.progress   out  fraction done, polled by MATLAB once a second

Input variables in retroLVtemp.mat:

    images          single/double, [frame, x, y, slice, dynamic]
    pixelSpacing    [dx dy] in mm, the in-plane size of one pixel
    sliceThickness  mm, carried through to the output for volumetry

Dynamics are segmented independently of one another: each is its own cardiac
cycle, so the anatomical clean-up derives its ventricle axis per dynamic and
end diastole and end systole are chosen per dynamic further downstream.

Output variables in retroLVtemp_SEG.mat:

    labels          uint8, [frame, x, y, slice, dynamic], 0/1/2 as the model
                    defines
    labelCounts     [frame, slice, 3, dynamic] voxel count per class, so that
                    MATLAB can do volumetry without recounting the label maps
    pixelSpacing    echoed back
    sliceThickness  echoed back
    cleanedSlices   number of slices emptied by the anatomical rules
    deblobbedSlices number of slices where a spurious component was removed

The third argument switches the anatomical clean-up in postprocess.py on or
off, so a study can be looked at both ways without changing code. The rules
remove components that are not the left ventricle and clear slices where the
ventricle was not found at all, which is mostly what goes wrong near the base
and the apex.

The clean-up runs on the model grid, before the labels are put back on the
image grid. Its threshold is in pixels and was calibrated at 0.13672 mm/pixel;
applying it after resampling would silently change what those pixels mean.

The two resampling steps are the reason this file exists at all. The model was
fine-tuned at a fixed 0.13672 mm/pixel on a 256 x 256 matrix and sees pixels,
not millimetres, so the images have to be brought onto that grid first. The
labels are then put back on the original grid with nearest neighbour
interpolation, because an interpolated label map is meaningless: averaging
class 0 and class 2 would produce class 1, that is, myocardium out of thin air.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from scipy.io import loadmat, savemat
from scipy.ndimage import zoom

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

INPUT_NAME = "retroLVtemp.mat"
OUTPUT_NAME = "retroLVtemp_SEG.mat"
PROGRESS_NAME = "retroLVseg.progress"

MODEL_SIZE = 256
NUM_CLASSES = 3

# The grid the model was fine-tuned on. Kept here because the ONNX graph carries
# no such metadata, and lv_segment.py, where they came from, cannot be imported
# without TensorFlow.
PIXEL_SPACING_MM = 0.13672      # the spacing the model was fine-tuned at

ONNX_NAME = "lv_segment.onnx"

# Mirrors postprocess.MAX_DIST_PX. Repeated rather than imported so that the
# default can be reported before postprocess is imported at all.
MAX_DIST_PX_DEFAULT = 30.0


def normalize(image):
    """Per-image min-max scaling to [0, 1]; a flat image maps to zeros.

    The same scaling lv_segment.normalize applies, repeated here so that nothing
    in this file needs importing that module.
    """
    a = np.asarray(image, dtype=np.float32)
    lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)


def load_segmenter(folder):
    """Return a function that turns a stack of images into label maps.

    The model is an ONNX graph, run through onnxruntime. It was converted from
    the Keras model the study produced and gives pixel-identical labels, which
    means the app needs neither TensorFlow nor its CUDA libraries: those pin
    against the ones torch needs for the reconstruction, and the two cannot both
    be satisfied in one environment.

    Returns (predict, description).
    """
    onnxPath = os.path.join(folder, "models", ONNX_NAME)

    if not os.path.isfile(onnxPath):
        raise FileNotFoundError("segmentation model not found: %s" % onnxPath)

    import onnxruntime as ort

    # Ask for an accelerator if this build has one, and let onnxruntime fall back
    # to the CPU by itself when it does not
    preferred = [p for p in ("CUDAExecutionProvider", "CoreMLExecutionProvider",
                             "CPUExecutionProvider")
                 if p in ort.get_available_providers()]

    session = ort.InferenceSession(onnxPath, providers=preferred)
    inputName = session.get_inputs()[0].name
    used = session.get_providers()[0] if session.get_providers() else "unknown"

    def predict(images):
        batch = np.stack([normalize(im) for im in np.asarray(images)])[..., None]
        # The network has five deep supervision heads; inference uses the last,
        # as the authors do
        result = session.run(None, {inputName: batch.astype(np.float32)})
        return np.argmax(result[-1], -1).astype(np.uint8)

    return predict, "onnxruntime (%s), %s" % (used, ONNX_NAME)


def write_progress(path, fraction):
    """Report progress to the app.

    Written to a temporary file and renamed, because os.replace is atomic and
    MATLAB polls this file once a second: it must never read a half written
    value. Progress reporting is cosmetic, so any failure is swallowed rather
    than allowed to end the segmentation.
    """
    try:
        temporary = path + ".tmp"
        with open(temporary, "w") as handle:
            handle.write("%.4f\n" % min(1.0, max(0.0, fraction)))
        os.replace(temporary, path)
    except OSError:
        pass


def fit_to_size(image, size):
    """Centre pad or crop a 2D array to size x size.

    Returns the array together with the offsets needed to undo it. A positive
    offset means rows or columns were cropped away, a negative one means they
    were padded on.
    """
    out = image
    offsets = []

    for axis in (0, 1):
        have = out.shape[axis]
        difference = have - size
        before = abs(difference) // 2

        if difference > 0:
            index = [slice(None), slice(None)]
            index[axis] = slice(before, before + size)
            out = out[tuple(index)]
        elif difference < 0:
            width = [(0, 0), (0, 0)]
            width[axis] = (before, -difference - before)
            out = np.pad(out, width, mode="constant")

        offsets.append((difference, before))

    return out, offsets


def undo_fit(labels, offsets, shape):
    """Inverse of fit_to_size, back onto an array of the given 2D shape."""
    out = labels

    for axis in (0, 1):
        difference, before = offsets[axis]

        if difference > 0:
            # rows or columns had been cropped, pad them back as background
            width = [(0, 0), (0, 0)]
            width[axis] = (before, difference - before)
            out = np.pad(out, width, mode="constant")
        elif difference < 0:
            # rows or columns had been padded, take them off again
            index = [slice(None), slice(None)]
            index[axis] = slice(before, before + shape[axis])
            out = out[tuple(index)]

    return out


def to_model_grid(image, spacing, target_spacing):
    """Resample one image onto the grid the model was fine-tuned at."""
    factors = (spacing[0] / target_spacing, spacing[1] / target_spacing)

    # A factor of exactly 1 still costs a copy through zoom, so skip it
    if abs(factors[0] - 1.0) < 1e-6 and abs(factors[1] - 1.0) < 1e-6:
        resampled = np.asarray(image, dtype=np.float32)
    else:
        resampled = zoom(np.asarray(image, dtype=np.float32), factors, order=1)

    fitted, offsets = fit_to_size(resampled, MODEL_SIZE)
    return fitted, offsets, resampled.shape


def from_model_grid(labels, offsets, resampled_shape, original_shape):
    """Put one label map back on the original image grid."""
    restored = undo_fit(labels, offsets, resampled_shape)

    if restored.shape == original_shape:
        return restored.astype(np.uint8)

    factors = (original_shape[0] / restored.shape[0],
               original_shape[1] / restored.shape[1])

    # order=0, nearest neighbour: interpolating between class labels would
    # invent classes that the model never predicted
    resized = zoom(restored, factors, order=0)

    # zoom can land one pixel short or long from rounding, so force the shape
    fixed, _ = fit_to_size(resized, max(original_shape))
    return fixed[:original_shape[0], :original_shape[1]].astype(np.uint8)


def main():
    data_folder = sys.argv[1]
    output_folder = sys.argv[2]

    # Anatomical clean-up on unless explicitly switched off. The study's numbers
    # were produced with it on, so that is the honest default.
    do_clean = True
    if len(sys.argv) > 3:
        do_clean = sys.argv[3].strip().lower() in ("1", "true", "yes", "clean", "on")

    max_dist_px = MAX_DIST_PX_DEFAULT
    if len(sys.argv) > 4 and sys.argv[4].strip():
        max_dist_px = float(sys.argv[4])

    progress_path = os.path.join(output_folder, PROGRESS_NAME)
    write_progress(progress_path, 0.0)


    contents = loadmat(os.path.join(data_folder, INPUT_NAME))
    images = contents["images"]
    spacing = np.asarray(contents["pixelSpacing"], dtype=float).ravel()
    slice_thickness = float(np.asarray(contents["sliceThickness"]).ravel()[0])

    # MATLAB drops trailing singleton dimensions, so a single slice or single
    # dynamic study arrives with one or two axes fewer than the full
    # [frame, x, y, slice, dynamic] layout.
    while images.ndim < 5:
        images = images[..., None]

    n_frames, n_x, n_y, n_slices, n_dynamics = images.shape
    print("Images: %d frames x %d slices x %d dynamic(s), %d x %d pixels at %.5f x %.5f mm"
          % (n_frames, n_slices, n_dynamics, n_x, n_y, spacing[0], spacing[1]), flush=True)
    print("Model grid: %d x %d at %.5f mm" % (MODEL_SIZE, MODEL_SIZE, PIXEL_SPACING_MM),
          flush=True)

    # Matrix size and field of view vary between acquisitions, and the resampling
    # above absorbs that. What it cannot absorb is a field of view larger than the
    # window the model works in: the extra is cropped away, centred, so anything
    # off-centre can be cut. Report it rather than let it happen quietly.
    window_mm = MODEL_SIZE * PIXEL_SPACING_MM
    fov_x_mm = n_x * spacing[0]
    fov_y_mm = n_y * spacing[1]
    print("Field of view: %.1f x %.1f mm, model window %.1f x %.1f mm"
          % (fov_x_mm, fov_y_mm, window_mm, window_mm), flush=True)

    if fov_x_mm > window_mm + 0.05 or fov_y_mm > window_mm + 0.05:
        print("WARNING: field of view exceeds the model window by %.1f x %.1f mm; "
              "the edges are cropped and an off-centre heart can be cut"
              % (max(0.0, fov_x_mm - window_mm), max(0.0, fov_y_mm - window_mm)),
              flush=True)

    if do_clean and n_slices == 1:
        print("NOTE: one slice only. The clean-up derives the ventricle axis from the "
              "slices themselves, so with a single slice the axis is that slice's own "
              "centroid: the 'is the ventricle here at all' rule cannot fire, and only "
              "the spurious component rule does any work.", flush=True)

    # Chosen here, not at module scope: loading a backend is what pulls in the
    # heavy imports, and a bad command line should be reported before that cost
    segmenter, backend = load_segmenter(HERE)
    print("Backend: %s" % backend, flush=True)
    print("Model loaded, segmenting ...", flush=True)

    labels = np.zeros(images.shape, dtype=np.uint8)
    counts = np.zeros((n_frames, n_slices, NUM_CLASSES, n_dynamics), dtype=np.int64)

    n_cleared = 0
    n_deblobbed = 0
    total_steps = float(n_dynamics * n_frames)
    step = 0

    if do_clean:
        from postprocess import clean_volume

    # Each dynamic is a cardiac cycle of its own, so it is segmented and cleaned
    # independently: the clean-up derives the ventricle axis per phase within a
    # dynamic, and borrowing that across dynamics would defeat the point of it.
    for dyn in range(n_dynamics):

        # The predictions are kept on the model grid until every frame of this
        # dynamic is done, both because the clean-up rules need the whole stack at
        # once and because their distance threshold is in pixels at
        # PIXEL_SPACING_MM. clean_volume wants (x, y, slice, phase).
        model_labels = np.zeros((MODEL_SIZE, MODEL_SIZE, n_slices, n_frames), dtype=np.uint8)
        geometry = {}

        # One batch per cardiac frame, over all slices, which is how the whole
        # study was segmented when the model was evaluated.
        for frame in range(n_frames):

            prepared = []

            for sl in range(n_slices):
                fitted, offsets, resampled_shape = to_model_grid(
                    images[frame, :, :, sl, dyn], spacing, PIXEL_SPACING_MM)
                prepared.append(fitted)
                geometry[(frame, sl)] = (offsets, resampled_shape)

            predicted = segmenter(np.stack(prepared))

            for sl in range(n_slices):
                model_labels[:, :, sl, frame] = predicted[sl]

            # The last tenth is left for the clean-up and the resampling back
            step += 1
            write_progress(progress_path, 0.9 * step / total_steps)

        if do_clean:
            model_labels, cleared, deblobbed = clean_volume(model_labels, max_dist_px)
            n_cleared += cleared
            n_deblobbed += deblobbed

        # Back onto the image grid, and count the classes after any clean-up
        for frame in range(n_frames):
            for sl in range(n_slices):
                offsets, resampled_shape = geometry[(frame, sl)]
                labels[frame, :, :, sl, dyn] = from_model_grid(
                    model_labels[:, :, sl, frame], offsets, resampled_shape, (n_x, n_y))

                for cls in range(NUM_CLASSES):
                    counts[frame, sl, cls, dyn] = int((model_labels[:, :, sl, frame] == cls).sum())

    if do_clean:
        print("Anatomical clean-up: %d slice(s) cleared, %d slice(s) deblobbed "
              "(threshold %.1f px = %.2f mm)"
              % (n_cleared, n_deblobbed, max_dist_px, max_dist_px * PIXEL_SPACING_MM),
              flush=True)
    else:
        print("Anatomical clean-up skipped, raw network output", flush=True)

    savemat(os.path.join(output_folder, OUTPUT_NAME),
            {"labels": labels,
             "labelCounts": counts,
             "pixelSpacing": spacing,
             "sliceThickness": slice_thickness,
             "cleanedSlices": n_cleared,
             "deblobbedSlices": n_deblobbed,
             # labelCounts is counted on the model grid, so volumetry must use the
             # model's pixel size and not the acquisition's. Exported rather than
             # repeated in MATLAB, so the two cannot drift apart.
             "modelPixelSpacing": PIXEL_SPACING_MM},
            do_compression=True)

    write_progress(progress_path, 1.0)
    print("Wrote %s" % OUTPUT_NAME, flush=True)


if __name__ == "__main__":
    main()
