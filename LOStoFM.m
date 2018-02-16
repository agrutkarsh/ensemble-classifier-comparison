function FM = LOStoFM( LOS )
% Converts a vector representing linear order statistic weights to a fuzzy
% measure. If LOS is length N, the resulting FM vector will be length
% 2^N - 1 in lexicographic order.
% Example:
% [0.5 0.5 0.75 0.5 0.75 0.75 1] = LOStoFM( [0.5 0.25 0.25] );

LOS = LOS(:);

N = length( LOS );
FM = zeros( 2^N - 1, 1 );

x = cumsum( LOS );

for cardinality = 1:N
    [inds, ~] = cardinalityMask( cardinality, N );
    FM( inds ) = x( cardinality );
end

function [inds, X] = cardinalityMask( cardinality, Nsingletons )
%
% Nsingletons = 4;
% cardinality = 2;

indlen = 2^Nsingletons - 1;

X = cell( indlen,1 );
inds = [];

for aa = 1:indlen
    
    % set 1s
    if mod( aa, 2 ) %(odd)
        X{ aa } = [X{ aa } 1];
    end
    
    % set 2s    
    if mod( aa, 2^2 ) > 1
        X{ aa } = [X{ aa } 2];
    end
    
    for bb = 3:Nsingletons
        if mod( aa, 2^bb ) > ((2^bb)/2 - 1)
            X{ aa } = [X{ aa } bb];
        end
    end
    
%     % set 3s
%     if mod( aa, 2^3 ) > 3
%         X{ aa } = [X{ aa } 3];
%     end
%     
%     % set 4s
%     if mod( aa, 2^4 ) > 7
%         X{ aa } = [X{ aa } 4];
%     end
%     
%     % set 5s
%     if mod( aa, 2^5 ) > 15
%         X{ aa } = [X{ aa } 5];
%     end
    
    if length( X{ aa } ) == cardinality
        inds = [inds aa];
    end

end