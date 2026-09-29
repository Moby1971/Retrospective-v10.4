% Where the four orientation labels go on the movie image as it is drawn
%
% Author : Gustav Strijkers
% Date   : 2026-09-28

function positions = orientationLabelPositions(inner, limits, aspect, yReversed, extent, letters, labelSize, fontSize)

    % Purpose:
    %   Return the positions of the four orientation labels that put each
    %   letter centred on its own edge of the image as the axes draw it, the
    %   same distance inside it on every side, whatever the image's shape, the
    %   zoom or the rotation.
    %
    % How it works:
    %   The axes draw their plot box inside their inner position, as large as
    %   the data aspect ratio allows and centred in it. In data units the box
    %   spans the limits; divided by the aspect ratio these give its shape, and
    %   the smaller of the two fits scales it to pixels. The part of the image
    %   inside the limits is mapped onto the box, with y downwards when the axes
    %   are reversed, as imshow sets them.
    %
    %   Each label is centred on the middle of its edge of that rectangle and
    %   moved inwards by the gap plus half its letter: half the letter's height
    %   for the top and bottom labels, half its width for the side ones. A label
    %   draws its text in its middle (HorizontalAlignment and VerticalAlignment
    %   'center'), so its centre is the letter's. The letter's size follows the
    %   font size: the cap height and the widths of Helvetica's capitals, the
    %   font the labels use.
    %
    % Inputs:
    %   inner     - [x y w h], the axes' InnerPosition in pixels of the parent
    %               the labels share with the axes
    %   limits    - [xmin xmax ymin ymax], the axes' limits
    %   aspect    - the axes' DataAspectRatio
    %   yReversed - true when y points down (YDir 'reverse')
    %   extent    - [x0 x1 y0 y1], the image's extent in data units, its
    %               outer pixel edges
    %   letters   - 1x4 cell of char, the right, top, left and bottom letters
    %   labelSize - [w h] of the labels
    %   fontSize  - the labels' font size, in pixels
    %
    % Output:
    %   positions - 4x4, a row [x y w h] per label in the order of letters, in
    %               the pixels of inner; rows of NaN when no part of the image
    %               is inside the limits

    gap = 4;                                    % pixels between a letter and its edge
    capHeight = 0.72 * fontSize;

    positions = nan(4, 4);

    % The plot box, fitted into the inner position with the data aspect ratio
    spanX = (limits(2) - limits(1)) / aspect(1);
    spanY = (limits(4) - limits(3)) / aspect(2);
    scale = min(inner(3) / spanX, inner(4) / spanY);
    boxW = spanX * scale;
    boxH = spanY * scale;
    boxX = inner(1) + (inner(3) - boxW) / 2;
    boxY = inner(2) + (inner(4) - boxH) / 2;

    % The part of the image inside the limits, in data units
    x0 = max(extent(1), limits(1));
    x1 = min(extent(2), limits(2));
    y0 = max(extent(3), limits(3));
    y1 = min(extent(4), limits(4));
    if x0 >= x1 || y0 >= y1
        return
    end

    % ... in pixels, y upwards as positions count it
    toX = @(x) boxX + (x - limits(1)) / (limits(2) - limits(1)) * boxW;
    if yReversed
        toY = @(y) boxY + boxH - (y - limits(3)) / (limits(4) - limits(3)) * boxH;
    else
        toY = @(y) boxY + (y - limits(3)) / (limits(4) - limits(3)) * boxH;
    end
    left = toX(x0);
    right = toX(x1);
    top = max(toY(y0), toY(y1));
    bottom = min(toY(y0), toY(y1));
    middleX = (left + right) / 2;
    middleY = (top + bottom) / 2;

    for k = 1:4
        halfWidth = letterWidth(letters{k}) * fontSize / 2;
        halfHeight = capHeight / 2;
        switch k
            case 1
                centre = [right - gap - halfWidth, middleY];
            case 2
                centre = [middleX, top - gap - halfHeight];
            case 3
                centre = [left + gap + halfWidth, middleY];
            otherwise
                centre = [middleX, bottom + gap + halfHeight];
        end
        positions(k, :) = [centre - labelSize / 2, labelSize];
    end

end % orientationLabelPositions


function width = letterWidth(letter)

    % The width of a capital in Helvetica, as a fraction of the font size: the
    % letters of the six directions and of head and feet. Another text is taken
    % at the width of its widest letter, an empty one (or spaces) as nothing.

    widths = struct('R', 0.722, 'L', 0.556, 'A', 0.667, 'P', 0.667, 'S', 0.667, ...
        'I', 0.278, 'H', 0.722, 'F', 0.611);
    width = 0;
    for c = upper(strtrim(char(letter)))
        if isfield(widths, c)
            width = max(width, widths.(c));
        else
            width = max(width, 0.722);
        end
    end

end % letterWidth
