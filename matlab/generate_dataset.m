function generate_dataset(preset, varargin)
%GENERATE_DATASET  Simulate fault records on the IEEE 14-bus model (ieee14_fault.slx).
%   generate_dataset('paper')   197 runs: Rf = Rg = 40 ohm, inception at t = 0.05 s, base load
%                                (replica of the original paper's dataset)
%   generate_dataset('varied')  4000 runs: Rf 0.5-100 ohm (log-uniform), random inception
%                                phase, load 70-130 %  (fixed seed -> reproducible)
%   Options: 'Workers' (default 8), 'OutDir' (default ../data/raw/<preset>), 'Model'.
%   Existing files are skipped, so an interrupted run can simply be restarted.
p = inputParser;
addRequired(p, 'preset', @(s) any(strcmp(s, {'paper', 'varied'})));
addParameter(p, 'Workers', 6);
addParameter(p, 'OutDir', '');
addParameter(p, 'Model', 'ieee14_fault');
parse(p, preset, varargin{:});  opt = p.Results;  mdl = opt.Model;
outdir = opt.OutDir;
if isempty(outdir), outdir = fullfile(fileparts(mfilename('fullpath')), '..', 'data', 'raw', preset); end
if ~isfolder(outdir), mkdir(outdir); end
load_system(mdl);

% ---- recording window -------------------------------------------------------
Ts    = 1.3e-6;              % powergui sample time (as in the paper)
f0    = 60;
nPre  = 2000;                % samples before inception in the paper-style window
W     = 12800;               % paper-style window length (~1 cycle)
LAG   = round(1/(f0*Ts));    % samples per cycle (12821) -> previous-cycle window
NREC  = LAG + W;             % recorded samples per run (25621)
NEVER = '[100]';             % switching time beyond the end of every simulation

% ---- fault classes: {type index, phases, grounded, HIF} ----------------------
F = {1,'A',1,0; 2,'B',1,0; 3,'C',1,0; 4,'AB',0,0; 5,'BC',0,0; 6,'AC',0,0;
     7,'AB',1,0; 8,'BC',1,0; 9,'AC',1,0; 10,'ABC',0,0; 11,'ABC',1,0;
     12,'A',0,1; 13,'B',0,1; 14,'C',0,1};

% ---- runs [type bus variant] and operating conditions [Rf ang lf] ------------
if strcmp(preset, 'paper'), NVAR = 1; NNF = 1; else, NVAR = 20; NNF = 80; rng(2027); end
runs = [zeros(NNF,1) zeros(NNF,1) (1:NNF)'];
for b = 1:14, for k = 1:14, for v = 1:NVAR, runs(end+1,:) = [k b v]; end, end, end %#ok<AGROW>
n = size(runs, 1);
if strcmp(preset, 'paper')
    cond = repmat([40 0 1], n, 1);
else
    cond = [0.5 * 10.^(rand(n,1)*log10(200)), ...   % Rf: 0.5-100 ohm, log-uniform
            rand(n,1)*360, ...                      % inception phase (deg)
            0.7 + 0.6*rand(n,1)];                   % load factor 0.7-1.3
end

% ---- clean base model: every fault idle -------------------------------------
for b = 1:14
    set_param(sprintf('%s/Fault_B%d', mdl, b), 'SwitchTimes', NEVER);
    for ph = 'ABC', set_param(sprintf('%s/HIF_B%d/Brk%s', mdl, b, ph), 'SwitchingTimes', NEVER); end
end
set_param(mdl, 'UnconnectedOutputMsg', 'none');
save_system(mdl);

L   = find_system(mdl, 'LookUnderMasks','all', 'FollowLinks','on', 'MaskType','Three-Phase Series RLC Load');
P0  = cellfun(@(x) str2double(get_param(x, 'ActivePower')),     L);
QL0 = cellfun(@(x) str2double(get_param(x, 'InductivePower')),  L);
QC0 = cellfun(@(x) str2double(get_param(x, 'CapacitivePower')), L);
oo  = {'off', 'on'};

% ---- one SimulationInput per missing run ------------------------------------
in = Simulink.SimulationInput.empty;
for r = 1:n
    ty = runs(r,1); b = runs(r,2); v = runs(r,3);
    Rf = cond(r,1); ang = cond(r,2); lf = cond(r,3);
    fname = fullfile(outdir, sprintf('%02d_%02d_%03d.mat', ty, b, v));
    if isfile(fname), continue; end

    tF = 0.05 + ang/360/f0;                      % fault inception
    tA = tF - (nPre + LAG)*Ts;                   % record start (one cycle before the window)
    si = Simulink.SimulationInput(mdl);
    si = si.setModelParameter('StopTime', sprintf('%.6f', tA + NREC*Ts + 1e-4));
    for k = 1:numel(L)
        si = si.setBlockParameter(L{k}, 'ActivePower',     sprintf('%.6g', lf*P0(k)));
        si = si.setBlockParameter(L{k}, 'InductivePower',  sprintf('%.6g', lf*QL0(k)));
        si = si.setBlockParameter(L{k}, 'CapacitivePower', sprintf('%.6g', lf*QC0(k)));
    end
    if ty > 0
        ph = F{ty,2}; gnd = F{ty,3}; hif = F{ty,4};
        if hif
            si = si.setBlockParameter(sprintf('%s/HIF_B%d/Brk%s', mdl, b, ph), ...
                                      'SwitchingTimes', sprintf('[%.9g]', tF));
        else
            fb = sprintf('%s/Fault_B%d', mdl, b);
            si = si.setBlockParameter(fb, 'FaultA', oo{any(ph=='A')+1});
            si = si.setBlockParameter(fb, 'FaultB', oo{any(ph=='B')+1});
            si = si.setBlockParameter(fb, 'FaultC', oo{any(ph=='C')+1});
            si = si.setBlockParameter(fb, 'GroundFault', oo{gnd+1});
            si = si.setBlockParameter(fb, 'FaultResistance',  sprintf('%.6g', Rf));
            si = si.setBlockParameter(fb, 'GroundResistance', sprintf('%.6g', Rf));
            si = si.setBlockParameter(fb, 'SwitchTimes',      sprintf('[%.9g]', tF));
        end
    end
    si = si.setPostSimFcn(@(out) save_run(out, fname, tA, Ts, NREC, LAG, nPre, ty, b, Rf, ang, lf));
    in(end+1) = si; %#ok<AGROW>
end

fprintf('%s: %d of %d runs to simulate -> %s\n', preset, numel(in), n, outdir);
if isempty(in), return; end
if isempty(gcp('nocreate')), parpool('Processes', opt.Workers); end
res = parsim(in, 'ShowProgress','on', 'StopOnError','off', ...
             'TransferBaseWorkspaceVariables','off', 'AttachedFiles', {which('save_run')});
bad = arrayfun(@(x) ~isempty(x.ErrorMessage), res);
fprintf('Done: %d failed (rerun to retry).\n', nnz(bad));
if any(bad), disp(res(find(bad, 1)).ErrorMessage); end
end

%[appendix]{"version":"1.0"}
%---
