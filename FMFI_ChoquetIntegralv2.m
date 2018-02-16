function [ result, SortInd ] = FMFI_ChoquetIntegral( confidences , measure )
%Discrete Choquet integral
% example -> ChoquetIntegral( confidences , measure )
%   [ result ] = ChoquetIntegral( [ 0.5 0.6 0.9 ] , FM )
% make sure measure is sized [1 x 2^n-1]
    
    [n,d]=size(confidences);
    %Sort the inputs (first step is sorting unless you use Mobius transform)
    [SortVal, SortInd] = sort( confidences , 2, 'descend' ); 
    
    %Append a 0 for the difference calculation below
    SortVal = [SortVal zeros(n,1)];
    
    i = cumsum(2.^(SortInd-1),2);
    result = sum(measure(i).*(SortVal(:,1:end-1)-SortVal(:,2:end)),2);
end