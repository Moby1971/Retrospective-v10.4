% 1D variable density filling cartesian k-space trajectory
% for Retrospective 
%
% LUT file for pe1_order = 3 (LUT) and retrospective  
% copy and paste the LUT into the no_views table in preclinical
%
%
% Gustav Strijkers
% Amsterdam UMC
% g.j.strijkers@amsterdamumc.nl
% July 2025
%


clearvars;
close all;
clc;



%% Define dimensions, use even numbers

targetKspaceSize = 192;                 % Size of resulting k-space                 192
trajectoryLength = 256;                 % Nr of views 2                             256
densityShape = "gauss";                 % Currently only "gauss" implemented 
sigma = 8;                              % Width of the more densily filled center   8
showPlot = true;                        % Show plot true/false
output = './output/';                   % Output directory


%% Checks

mustBePosEvenInt(targetKspaceSize);
mustBePosEvenInt(trajectoryLength);
mustBePosEvenInt(sigma);



%% Calculate target k-space

kSpaceCenter = floor(targetKspaceSize/2 + 1);
extraLines = trajectoryLength-targetKspaceSize;
k = 1:targetKspaceSize;

switch densityShape

    case "gauss"

        % Gaussian function
        d = (1/(sigma*sqrt(2*pi)))*exp(-((k-kSpaceCenter).^2)./(2*sigma.^2));

end

% Make discrete
incr = 0.9;
df = round(d*incr);
while sum(df) <= extraLines
    df = round(d*incr);
    incr = incr + 0.001;
end

% Make sure that total number of k-lines equals the trajectory length
pm = true;
while sum(df) > extraLines
    loc = find(df==1);
    if pm
        df(loc(1)) = 0;
    else
        df(loc(end)) = 0;
    end
    pm = ~pm;
end
d = 1 + df;




%% Construct the zigzag trajectory

cnt = 1;
traj = zeros(targetKspaceSize,1);

for k = 1:2:targetKspaceSize

    for c = 1:d(k)
        traj(cnt) = k;
        cnt = cnt + 1;
    end

end

for k = targetKspaceSize:-2:2

    for c = 1:d(k)
        traj(cnt) = k;
        cnt = cnt + 1;
    end

end

traj = traj - targetKspaceSize/2 - 1;




%% Simulate a random k-space filling

N = 240000;

samples = randi([1 trajectoryLength], N, 1);

filling = zeros(targetKspaceSize,1);

for cnt = 1:length(samples)

    index = traj(samples(cnt)) + targetKspaceSize/2 + 1;
    filling(index) = filling(index) + 1;

end

filling = filling / sum(filling);



%% Plot the result

if showPlot

    titleFontSize = 28;
    axisLabelFontSize = 20;
    axisFontSize = 16;
    lineWidth = 3;

    fig = figure(11);
    fig.Position(3:4) = [1200,500];
    t = tiledlayout(1,2);

    nexttile;
    hold on;
    plot(traj,'LineWidth', lineWidth);
    box on;
    title("Trajectory",'FontSize', titleFontSize);
    axis([0 trajectoryLength+1 min(traj)*1.1 max(traj)*1.1 ]);
    xlabel("Sample",'FontSize',axisLabelFontSize,'FontName','Arial');
    ylabel("K-line",'FontSize',axisLabelFontSize,'FontName','Arial');
    ax = gca;
    set(ax,'FontSize',axisFontSize,'FontName','Arial')
    hold off;

    nexttile;
    hold on;
    plot(filling,'LineWidth', lineWidth);
    box on;
    title("Estimated filling of k-space",'FontSize', titleFontSize);
    axis([0 targetKspaceSize+1 0 max(filling)*1.1 ]);
    xlabel("K-line",'FontSize',axisLabelFontSize,'FontName','Arial');
    ylabel("Filling density",'FontSize',axisLabelFontSize,'FontName','Arial');
    ax = gca;
    set(ax,'FontSize',axisFontSize,'FontName','Arial')
    hold off;

end


%% Export the trajectory


filename = strcat(output,filesep,'LUT_Cartesian_1D_',num2str(targetKspaceSize),'_',num2str(trajectoryLength),'_',num2str(sigma),'.txt');
fileID = fopen(filename,'w');

for i = 1:length(traj)

    fprintf(fileID,num2str(traj(i)));
    fprintf(fileID,',\n');

end

fclose(fileID);




%% Summary of k-space filling

centerWidth = 0.2 * targetKspaceSize;  % central 20%
centerIdx = round(targetKspaceSize/2 - centerWidth/2 + 1) : round(targetKspaceSize/2 + centerWidth/2);
edgeIdx = [1:round(0.1*targetKspaceSize), round(0.9*targetKspaceSize):targetKspaceSize];

fprintf('\n--- K-space filling summary ---\n');
fprintf('Target k-space size    : %d\n', targetKspaceSize);
fprintf('Trajectory length      : %d\n', trajectoryLength);
fprintf('Density shape          : %s\n', densityShape);
fprintf('Sigma                  : %d\n', sigma);
fprintf('Mean filling           : %.4f\n', mean(filling));
fprintf('Min filling            : %.4f\n', min(filling));
fprintf('Max filling            : %.4f\n', max(filling));
fprintf('Std of filling         : %.4f\n', std(filling));
fprintf('Center/Edge fill ratio : %.2f\n', mean(filling(centerIdx)) / mean(filling(edgeIdx)));
fprintf('-------------------------------\n\n');





%% Helper function

function mustBePosEvenInt(x)

    % Throw an error unless x is:
    %   – a positive number
    %   – an integer (no fractional part)
    %   – even

    assert( isnumeric(x)             && ... % numeric input
            all(x > 0)               && ... % positive
            all(mod(x,1) == 0)       && ... % integer
            all(mod(x,2) == 0), ...         % even
            'Input must be a positive, even integer.');

end




