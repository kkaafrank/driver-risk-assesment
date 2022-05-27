imds = imageDatastore("C:\Users\Kevin\Desktop\Apps\Code\Python\Driver Identification and Classification\data\VA Minus 00 Images");
aimds = augmentedImageDatastore([227,227], imds);
layer = 'fc7';
net = alexnet;
features = activations(net, aimds, layer, 'OutputAs', 'rows');
[coeff, ~, ~, ~, exp, ~] = pca(features, 'Economy', false);
sum(exp(1:5))
top5pca = coeff(:,1:5);
reduced = mtimes(features, top5pca);
writematrix(reduced, 'C:\Users\Kevin\Desktop\Apps\Code\Python\Driver Identification and Classification\data\VelAng Subset\VelAng Subset PCA.csv');
save('VelAng Transfer Subset.mat');