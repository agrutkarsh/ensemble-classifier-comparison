function model = defimkltrain( training_label_vector,...
    training_instance_matrix, rbf_kernel_gamma_vector, reg_type, lambda)
    
if nargin < 4 %number of function input arguments
    reg_type = [];
end

if isempty( reg_type )
    reg_type = 'l1';
    lambda = 0;
end

y = training_label_vector;
X = training_instance_matrix;

number_of_kernels = length( rbf_kernel_gamma_vector ); %number of elements

N = number_of_kernels * (2 ^( number_of_kernels-1 )- 1); % # of constraints
g = 2^(number_of_kernels)-1; % measure length {g_1,g_2,...,g_12,...g_12..K}

% Get decision values from LIBSVM for each kernel
for k = 1 : number_of_kernels,
    svmmodel{k} = svmtrain(y,X,['-t 2 -g ',num2str(rbf_kernel_gamma_vector(k)),'-q']);
    [~,~,dvtrain(:,k)] = svmpredict(y,X,svmmodel{k},'-q');
end

dvtrain = dvtrain./sqrt(1+dvtrain.^2);

[C,D,f,~,~]=QPmatrices(number_of_kernels,dvtrain,y,[]);

options = optimset('Display','off');

if strcmp( reg_type, 'l1' ) % l1 regularization
    FM = quadprog(2*D,f+lambda,C,zeros(size(C,1),1),[],[],[zeros(g-1,1); 1],ones(g,1),0*ones(g,1),options);
    
elseif strcmp( reg_type, 'l2' ) % l2 regularization (goal = min)
    FM = quadprog((2*D+lambda*eye(g)),f,C,zeros(size(C,1),1),[],[],[zeros(g-1,1); 1],ones(g,1),[],options);
    
elseif strcmp( reg_type, 'max' ) % goal = max operator
    G = ones( size (f ) ); % (max aggregation)
    FM = quadprog((2*D+lambda*eye(g)),(f-2*lambda.*G),C,zeros(size(C,1),1),[],[],[zeros(g-1,1); 1],ones(g,1),[],options);
    
elseif strcmp( reg_type, 'mean' ) % goal = mean operator
    G = LOStoFM( ones( number_of_kernels, 1 ) / number_of_kernels );
    FM = quadprog((2*D+lambda*eye(g)),(f-2*lambda.*G),C,zeros(size(C,1),1),[],[],[zeros(g-1,1); 1],ones(g,1),[],options);    
    
else
    error('Invalid norm type');
end

model.FM = FM;
model.svmmodels = svmmodel;