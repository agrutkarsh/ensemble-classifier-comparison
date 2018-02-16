function model = defimkltrain_new( training_label_vector,...
    training_instance_matrix, rbf_kernel_gamma_vector, reg_type, lambda)

if nargin < 4
    reg_type = [];
end

if isempty( reg_type )
    reg_type = 'l1';
    lambda = 0;
end

y = training_label_vector;
X = training_instance_matrix;


    number_of_kernels = length( rbf_kernel_gamma_vector );

    N = number_of_kernels * (2 ^( number_of_kernels-1 )- 1); % # of constraints
    g = 2^(number_of_kernels)-1; % measure length {g_1,g_2,...,g_12,...g_12..K}
        for k = 1 : number_of_kernels,
            svmmodel{k} = svmtrain(y,X,['-t 2 -q -g ',num2str(rbf_kernel_gamma_vector(k)),'-q']);
            [~,~,dvtrain(:,k)] = svmpredict(y,X,svmmodel{k},'-q');
        end
        
dvtrain = dvtrain./sqrt(1+dvtrain.^2);
[C,D,f,~,~]=QPmatrices(number_of_kernels,dvtrain,y,[]);
options = optimset('Display','off');

if strcmp( reg_type, 'l1' ) || strcmp( reg_type, 'l1min' ) % l1 regularization on FM directly (equivalent to goal = min (l1) )
    [FM,FVAL] = quadprog(2*D,f+lambda,C,zeros(size(C,1),1),[],[],[zeros(g-1,1); 1],ones(g,1),0*ones(g,1),options);
    
elseif strcmp( reg_type, 'l2' ) || strcmp( reg_type, 'l2min' ) % l2 regularization on FM directly (equivalent to goal = min (l2) )
    FM = quadprog((2*D+lambda*eye(g)),f,C,zeros(size(C,1),1),[],[],[zeros(g-1,1); 1],ones(g,1),[],options);
    
elseif strcmp( reg_type, 'max' ) || strcmp( reg_type, 'l2max' )% goal = max operator (l2)
    G = ones( size (f ) ); % (max aggregation)
    FM = quadprog((2*D+lambda*eye(g)),(f-2*lambda.*G),C,zeros(size(C,1),1),[],[],[zeros(g-1,1); 1],ones(g,1),[],options);
    
elseif strcmp( reg_type, 'mean' ) || strcmp( reg_type, 'l2mean' ) % goal = mean operator (l2)
    G = LOStoFM( ones( number_of_kernels, 1 ) / number_of_kernels );
    FM = quadprog((2*D+lambda*eye(g)),(f-2*lambda.*G),C,zeros(size(C,1),1),[],[],[zeros(g-1,1); 1],ones(g,1),[],options);    
    
elseif strcmp( reg_type, 'l1max' ) % goal = max (l1)
    G = ones( size (f ) ); % (max aggregation)
    t = 1/lambda;
    
    Dx = [D zeros( size( D ) );zeros( size( D ) ) zeros( size( D ) )];
    fx = [f; zeros( size( f ) )];
    Cx = [C zeros( size( C ) )];
    Bl = [zeros(g-1,1); 1; G];
    Br = [ones(g,1); G];
    bineq = zeros(size(C,1),1);
    FM = quadprog(2*Dx,fx,Cx,bineq,[],[],Bl,Br,0*ones(g,1),options);
    u0 = FM(1:end/2);
    delta = sign(u0 - G); delta = [delta' -delta'];
    ii=0;
    while delta*FM > t
        ii = ii + 1;
        Cx = [Cx; delta]; 
        bineq = [bineq; t];
        FM = quadprog(2*Dx,fx,Cx,bineq,[],[],Bl,Br,0*ones(g,1),options);
        uu = FM(1:end/2);
        delta = sign(uu - G); delta = [delta' -delta'];
    end
    FM = FM(1:end/2);
    
elseif strcmp( reg_type, 'l1mean' ) % goal = mean (l1)
    G = LOStoFM( ones( number_of_kernels, 1 ) / number_of_kernels )';
    t = 1/lambda;
    
    Dx = [D zeros( size( D ) );zeros( size( D ) ) zeros( size( D ) )];
    fx = [f; zeros( size( f ) )];
    Cx = [C zeros( size( C ) )];
    Bl = [zeros(g-1,1); 1; G];
    Br = [ones(g,1); G];
    bineq = zeros(size(C,1),1);
    [FM,FVAL] = quadprog(2*Dx,fx,Cx,bineq,[],[],Bl,Br,0*ones(g,1),options);
    u0 = FM(1:end/2);
    delta = sign(u0 - G); delta = [delta' -delta'];
    ii=0;
    while delta*FM > t
        ii = ii + 1;
        Cx = [Cx; delta]; 
        bineq = [bineq; t];
        FM = quadprog(2*Dx,fx,Cx,bineq,[],[],Bl,Br,0*ones(g,1),options);
        uu = FM(1:end/2);
        delta = sign(uu - G); delta = [delta' -delta'];
    end
    FM = FM(1:end/2);
%     sprintf('Done in %i iterations.', ii)
%     pause(1)
    
elseif strcmp( reg_type, 'l1los' )
    loslength = number_of_kernels;
    t = 1/lambda;
    A = gimmeA(loslength);

    Dz = [D zeros(size(D,1),loslength);zeros(loslength,size(D,2)) zeros(loslength)];
    fz = [f; zeros(loslength,1)];
    Cz = [C zeros(size(C,1),loslength)];
    blz = [zeros(g-1,1); 1; -ones(loslength,1)];
    brz = ones(g+loslength,1);
    bineq = zeros(size(Cz,1),1);
    FM = quadprog(2*Dz,fz,Cz,bineq,[],[],blz,brz,0*ones(g,1),options);
    S = [eye(g) -A];
    delta = sign(S*FM); %delta = [delta; -delta];
    G = ones(1,g)*bsxfun(@times, delta, S);
    ii = 0;
    while G*FM > t
        ii = ii + 1;
        G = ones(1,g)*bsxfun(@times, delta, S);
        Cz = [Cz; G];
        bineq = [bineq; t];
        FM = quadprog(2*Dz,fz,Cz,bineq,[],[],blz,brz,0*ones(g,1),options);
        delta = sign(S*FM);
    end
    LOS = FM(g+1:end);
    FM = FM(1:g);
    
elseif strcmp( reg_type, 'l2los' )
    loslength = number_of_kernels;
    A = gimmeA(loslength);

    Dv = [(D+lambda*eye(size(D))) -lambda*A;-lambda*A' lambda*A'*A];
    fv = [f; zeros(loslength,1)];
    Cv = [C zeros(size(C,1),loslength)]; %assuming B isn't needed
    Blv = [zeros(g-1,1); 1; zeros(loslength,1)];
    Brv = ones(g+loslength,1);
    bineq = zeros(size(Cv,1),1);
    FM = quadprog(2*Dv,fv,Cv,bineq,[],[],Blv,Brv,0*ones(g,1),options);
    LOS = FM(g+1:end);
    FM = FM(1:g);
    model.LOS = LOS;
    
else
    error('Invalid regularization type');
end

model.FM = FM;
model.svmmodels = svmmodel;

function A = gimmeA( N )
% N - number of inputs to FM

A = zeros(2^N - 1,N);

for cardinality = 1:N
    [inds, ~] = cardinalityMask( cardinality, N );
    A( inds, cardinality ) = 1;
end

function [inds, X] = cardinalityMask( cardinality, Nsingletons )
%%
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