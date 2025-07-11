# 1D variable density filling cartesian k-space trajectory
# for retrospective
# 
# LUT file for pe1_order = 3 (LUT) and retrospective  
# copy and paste the LUT into the no_views table in preclinical
#
#
# Gustav Strijkers
# Amsterdam UMC
# g.j.strijkers@amsterdamumc.nl
# July 2025
#


import numpy as np
import matplotlib.pyplot as plt
import os

# Parameters

targetKspaceSize = 192         # Size of resulting k-space                 192
trajectoryLength = 256         # Nr of views 2                             256
densityShape = "gauss"         # Currently only "gauss" implemented
sigma = 8                      # Width of the more densily filled center   8
showPlot = False               # Show the plot True / False
output = './output/'           # Output folder

# Checks

def mustBePosEvenInt(x):
    assert isinstance(x, int) and x > 0 and x % 2 == 0, "Input must be a positive, even integer."

mustBePosEvenInt(targetKspaceSize)
mustBePosEvenInt(trajectoryLength)
mustBePosEvenInt(sigma)

# Calculate target k-space

kSpaceCenter = targetKspaceSize // 2
extraLines = trajectoryLength - targetKspaceSize
k = np.arange(1, targetKspaceSize + 1)

if densityShape == "gauss":
    d = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((k - (kSpaceCenter + 1)) ** 2) / (2 * sigma ** 2))
else:
    raise ValueError("Unsupported density shape")

# Make discrete
incr = 0.9
df = np.round(d * incr).astype(int)
while np.sum(df) <= extraLines:
    incr += 0.001
    df = np.round(d * incr).astype(int)

# Trim to match trajectory length
pm = True
while np.sum(df) > extraLines:
    loc = np.where(df == 1)[0]
    if len(loc) == 0:
        break
    if pm:
        df[loc[0]] = 0
    else:
        df[loc[-1]] = 0
    pm = not pm

d = 1 + df


# Construct the zigzag trajectory
traj = []
for i in range(0, targetKspaceSize, 2):
    traj.extend([i + 1] * d[i])

for i in range(targetKspaceSize - 1, 0, -2):
    traj.extend([i + 1] * d[i])

traj = np.array(traj) - targetKspaceSize // 2 - 1


# Simulate random k-space filling
N = 240000
samples = np.random.randint(0, trajectoryLength, N)
filling = np.zeros(targetKspaceSize, dtype=int)

for s in samples:
    index = traj[s] + targetKspaceSize // 2
    if 0 <= index < targetKspaceSize:
        filling[index] += 1

filling = filling / np.sum(filling)


# Plotting
if showPlot:
    titleFontSize = 18
    axisLabelFontSize = 14
    axisFontSize = 12
    lineWidth = 2

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].plot(traj, linewidth=lineWidth)
    axs[0].set_title("Trajectory", fontsize=titleFontSize)
    axs[0].set_xlabel("Sample", fontsize=axisLabelFontSize)
    axs[0].set_ylabel("K-line", fontsize=axisLabelFontSize)
    axs[0].tick_params(labelsize=axisFontSize)
    axs[0].grid(True)

    axs[1].plot(filling, linewidth=lineWidth)
    axs[1].set_title("Estimated filling of k-space", fontsize=titleFontSize)
    axs[1].set_xlabel("K-line", fontsize=axisLabelFontSize)
    axs[1].set_ylabel("Filling density", fontsize=axisLabelFontSize)
    axs[1].tick_params(labelsize=axisFontSize)
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()


# Export trajectory to text file
filename = os.path.join(output, f"LUT_cartesian_1D_{targetKspaceSize}_{trajectoryLength}_{sigma}.txt")
with open(filename, 'w') as f:
    for t in traj:
        f.write(f"{t},\n")


# Summary
centerWidth = int(0.2 * targetKspaceSize)
centerStart = targetKspaceSize // 2 - centerWidth // 2
centerEnd = centerStart + centerWidth
centerIdx = np.arange(centerStart, centerEnd)

edgeIdx = np.concatenate((np.arange(0, int(0.1 * targetKspaceSize)), 
                          np.arange(int(0.9 * targetKspaceSize), targetKspaceSize)))

ky_min = int(traj.min())
ky_max = int(traj.max())

print("\n--- K-space filling summary ---")
print(f"Target k-space size    : {targetKspaceSize}")
print(f"Trajectory length      : {trajectoryLength}")
print(f"Density shape          : {densityShape}")
print(f"Sigma                  : {sigma}")
print(f"Mean filling           : {np.mean(filling):.4f}")
print(f"Min filling            : {np.min(filling):.4f}")
print(f"Max filling            : {np.max(filling):.4f}")
print(f"Std of filling         : {np.std(filling):.4f}")
print(f"Center/Edge fill ratio : {np.mean(filling[centerIdx]) / np.mean(filling[edgeIdx]):.2f}")
print(f"ky range               : {ky_min} to {ky_max}")
print("-------------------------------\n")
