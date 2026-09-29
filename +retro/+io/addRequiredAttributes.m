% Add the attributes an MR image carries even when empty
%
% Author : Gustav Strijkers
% Date   : 2026-09-27

function header = addRequiredAttributes(header, moment)

    % Purpose:
    %   Give a DICOM header the attributes an MR image must carry, even empty, that
    %   it lacks: the content date and time, the accession number, the position
    %   reference indicator and the patient position.
    %
    % How it works:
    %   The exporters write their headers with dicomwrite in its Copy mode, which
    %   writes a header as it is given and adds nothing, so these are added here,
    %   with the values dicomwrite supplies when it completes a header itself: the
    %   date and time of the export for the content, empty for the other three. A
    %   value the header already has, empty or not, is kept.
    %
    %   These five are the ones missing from the headers the exporters start from.
    %   MR Solutions writes no content date and no position reference indicator,
    %   ParaVision 6 no content date and time, retro.io.exportDicomMat builds
    %   neither the content date, the accession number nor the position reference
    %   indicator, and both exporters set the patient position only when the
    %   subject angles name a placement.
    %
    % Inputs:
    %   header - struct, a DICOM header as dicominfo gives it or an exporter builds it
    %   moment - datetime, the moment of export
    %
    % Output:
    %   header - the header with the missing attributes added

    required = { ...
        'ContentDate', char(moment, 'yyyyMMdd'); ...
        'ContentTime', char(moment, 'HHmmss.SSSSSS'); ...
        'AccessionNumber', ''; ...
        'PositionReferenceIndicator', ''; ...
        'PatientPosition', ''};

    for k = 1:size(required, 1)
        if ~isfield(header, required{k, 1})
            header.(required{k, 1}) = required{k, 2};
        end
    end

end % addRequiredAttributes
