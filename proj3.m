function [Wlr,blr,Wnn1,Wnn2,bnn1,bnn2] = proj3()
%**********************************Loading the Data******************************%
%loadMNISTImages returns a 28x28x[number of MNIST images] matrix containing
%the raw MNIST images
filename = 'train-images.idx3-ubyte';
fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);
magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2051, ['Bad magic number in ', filename, '']);
trainNumImages = fread(fp, 1, 'int32', 0, 'ieee-be');
trainNumRows = fread(fp, 1, 'int32', 0, 'ieee-be');
trainNumCols = fread(fp, 1, 'int32', 0, 'ieee-be');
trainImages = fread(fp, inf, 'unsigned char');
trainImages = reshape(trainImages, trainNumCols, trainNumRows, trainNumImages);
%trainImages = permute(trainImages,[2 1 3]);
fclose(fp);
% Reshape to #pixels x #examples
trainImages = reshape(trainImages, size(trainImages, 1) * size(trainImages, 2), size(trainImages, 3));
% Convert to double and rescale to [0,1]
trainImages = double(trainImages) / 255;


filename = 't10k-images.idx3-ubyte';
fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);
magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2051, ['Bad magic number in ', filename, '']);
validNumImages = fread(fp, 1, 'int32', 0, 'ieee-be');
validNumRows = fread(fp, 1, 'int32', 0, 'ieee-be');
validNumCols = fread(fp, 1, 'int32', 0, 'ieee-be');
validImages = fread(fp, inf, 'unsigned char');
validImages = reshape(validImages, validNumCols, validNumRows, validNumImages);
%validImages = permute(validImages,[2 1 3]);
fclose(fp);
% Reshape to #pixels x #examples
validImages = reshape(validImages, size(validImages, 1) * size(validImages, 2), size(validImages, 3));
% Convert to double and rescale to [0,1]
validImages = double(validImages) / 255;



%loadMNISTLabels returns a [number of MNIST images]x1 matrix containing
%the labels for the MNIST images
filename = 'train-labels.idx1-ubyte';
fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);
magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2049, ['Bad magic number in ', filename, '']);
trainNumLabels = fread(fp, 1, 'int32', 0, 'ieee-be');
trainLabels = fread(fp, inf, 'unsigned char');
assert(size(trainLabels,1) == trainNumLabels, 'Mismatch in label count');
fclose(fp);

filename = 't10k-labels.idx1-ubyte';
fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);
magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2049, ['Bad magic number in ', filename, '']);
validNumLabels = fread(fp, 1, 'int32', 0, 'ieee-be');
validLabels = fread(fp, inf, 'unsigned char');
assert(size(validLabels,1) == validNumLabels, 'Mismatch in label count');
fclose(fp);


trainsize=60000;

validsize=10000;
validsize1=validsize+1;
K = 10;
trainTarget = zeros(trainsize,K);
validTarget = zeros(validsize,K);
for i=1:trainsize
  trainTarget(i,(trainLabels(i)+1)) = 1;
end 

for i=1:validsize
  validTarget(i,(validLabels(i)+1)) = 1;
end

Wlr = randn(785,10);

trainImages1 = ([ones(1,trainsize);trainImages(:,:)]);
validImages1 = ([ones(1,validsize);validImages(:,:)]);

eta =0.015;

for j=1:2
for i= 1:trainsize
    Wlr = Wlr - eta * transpose((softmax(transpose(Wlr)*trainImages1(:,i))-transpose(trainTarget(i,:))) * transpose(trainImages1(:,i)));

end
end





Nright=0;

outputValid=zeros(1,10);
%LOOP2

for i=1:validsize
   
   outputValid=softmax(transpose(Wlr)*validImages1(:,i));
   
   [X,Y]=max(outputValid);
   
   if (Y==find(validTarget(i,:)))
   %if (isequal(yk,TValid(i)) )
    Nright=Nright+1; 
   end

end



ErrorLR= (10000-Nright)/10000*100
AccuracyLR=Nright/10000*100

blr = Wlr(1,:);
Wlr = Wlr(2:length(Wlr(:,1)),:);



h='sigmoid';
D=784

J = D+1;

Wnn1 = randn(D+1,J); 
Wnn2 = randn(J+1,K);
eta1 = 0.053;
eta2 = 0.053;

for i= 1:trainsize
   
   Z = logsig(transpose(Wnn1) * trainImages1(:,i));
  
   Z1 = [1;Z(:,:)];
   Y = logsig(transpose(Wnn2) * Z1);
   dk = Y - transpose(trainTarget(i,:));
   dj = Z .* (1- Z) .* (Wnn2(2:length(Wnn2),:) * dk);
   Wnn1 = Wnn1 - transpose(eta1 * dj * transpose(trainImages1(:,i)));
   Wnn2 = Wnn2 - transpose(eta2 * dk * transpose(Z1));         
end

Nright1 = 0;
for i=1:validsize
    
    Z = logsig(transpose(Wnn1) * validImages1(:,i));
   
    Z1 = [1;Z(:,:)];
    Yk= logsig(transpose(Wnn2) * Z1);
    [X,Y]=max(abs(Yk));
   
    if (Y==find(validTarget(i,:)))
   %if (isequal(yk,TValid(i)) )
     Nright1=Nright1+1; 
    end

end
ErrorNN= (10000-Nright1)/10000*100
AccuracyNN=Nright1/10000*100

bnn1 = Wnn1(1,:);
Wnn1 = Wnn1(2:length(Wnn1(:,1)),:);

bnn2 = Wnn2(1,:);
Wnn2 = Wnn2(2:length(Wnn2(:,1)),:);

save proj3.mat
end

function g = logsig(z)
%SIGMOID Compute sigmoid function
%   J = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).


g = 1 ./ (1 + exp(-z));


% =============================================================

end

function g = softmax(z)
%SIGMOID Compute sigmoid function
%   J = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).


g = exp(z)/sum(exp(z));


% =============================================================

end

