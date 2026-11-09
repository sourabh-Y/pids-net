mdl = 'ieee14_fault'; load_system(mdl);
fclose('all');                                   % close the half-written file from the failed run
fid = fopen('model_report.txt','w');
fprintf(fid, '%s\n', evalc('ver'));
fprintf(fid, 'Solver: %s  FixedStep: %s\n', get_param(mdl,'Solver'), get_param(mdl,'FixedStep'));

types = {'PSB option menu block', 'Three-Phase Source', 'Three-Phase Transformer (Two Windings)', ...
         'Three-Phase Series RLC Load', 'Three-Phase Series RLC Branch', ...
         'Three-Phase VI Measurement', 'Three-Phase Fault'};
for t = 1:numel(types)
    B = find_system(mdl, 'LookUnderMasks','all', 'FollowLinks','on', 'MaskType',types{t});
    fprintf(fid, '\n===== %s (%d blocks) =====\n', types{t}, numel(B));
    dumpBlocks(fid, B);
end

fprintf(fid, '\n===== HIF_B1 internals =====\n');
dumpBlocks(fid, find_system([mdl '/HIF_B1'], 'LookUnderMasks','all', 'FollowLinks','on', ...
                            'Type','block', 'Mask','on'));

fprintf(fid, '\n===== To Workspace blocks =====\n');
B = find_system(mdl, 'SearchDepth',1, 'BlockType','ToWorkspace');
for k = 1:numel(B)
    fprintf(fid, '%s  var=%s  Ts=%s  fmt=%s\n', strrep(B{k}, newline, ' '), ...
        get_param(B{k},'VariableName'), get_param(B{k},'SampleTime'), get_param(B{k},'SaveFormat'));
end
fclose(fid); disp('Wrote model_report.txt');

function dumpBlocks(fid, B)
    for k = 1:numel(B)
        b  = B{k};
        dp = get_param(b, 'DialogParameters');
        if isstruct(dp), names = fieldnames(dp); else, names = {}; end
        vals = cellfun(@(p) sprintf('%s=%s', p, val2str(b, p)), names, 'UniformOutput', false);
        fprintf(fid, '%s\n  %s\n', strrep(b, newline, ' '), strjoin(vals, ' | '));
    end
end

function s = val2str(b, p)
    try, s = strtrim(char(string(get_param(b, p)))); catch, s = '?'; end
    s = strrep(s, newline, ' ');
end