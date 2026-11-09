function r = save_run(out, fname, tA, Ts, NREC, lag, nPre, ty, b, Rf, ang, lf)
%SAVE_RUN  PostSimFcn for parsim: store a 2-cycle record of the five monitoring points.
%   I, V : NREC x 15 single (A, V); columns [a b c] for monitoring points 1..5
%   lag  : samples per cycle (previous-cycle offset); nPre: samples before inception
%          in the paper-style window rec(lag+1 : lag+12800)
    I = zeros(NREC, 15, 'single');  V = zeros(NREC, 15, 'single');
    for m = 1:5
        si = out.get(sprintf('I%d', m));  k0 = find(si.time >= tA - Ts/2, 1);
        I(:, 3*m-2:3*m) = si.signals.values(k0:k0+NREC-1, :);
        sv = out.get(sprintf('V%d', m));  k0 = find(sv.time >= tA - Ts/2, 1);
        V(:, 3*m-2:3*m) = sv.signals.values(k0:k0+NREC-1, :);
    end
    typeIdx = ty; bus = b;
    save(fname, 'I', 'V', 'typeIdx', 'bus', 'Rf', 'ang', 'lf', 'lag', 'nPre');
    r.saved = fname;
end