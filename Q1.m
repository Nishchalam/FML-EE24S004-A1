% (1) You are given a dataset with 1000 data points each in R2 . % (a)
% Write a piece of code to run the PCA algorithm on this data-set. How much
% of % the variance in the data-set is explained by each of the principal
% components? % (b) Write a piece of code to implement the Kernel PCA
% algorithm on this dataset. % Explore various kernels discussed in class.
% For each Kernel, plot the projection % of each point in the dataset onto
% the top-2 principal components. Use one plot % for each kernel - In case
% of RBF kernel, use a different plot for each value of σ % that you use. %
% (c) Which Kernel do you think is best suited for this dataset and why?
clear all;
close all;

% User defined function to compute mean across columns of a mxn array
function average = compute_average(data)
    [m,n] = size(data);
    average = zeros(1,n); 
    for i=1:n
        col_sum = 0;
        for j=1:m
            col_sum = col_sum + data(j,i);
        end
        average(i) = col_sum / m;
    end
end
%User defined for computing covariance of data
function covariance_mat = compute_covariance(data)
    [m,n]=size(data);
    avg=compute_average(data);
    centered_data=data-avg;
    covariance_mat=(centered_data'*centered_data)/m;
end
%% (a) PCA
dataTable = readtable("dataset1-assignment1 - Sheet1.csv");
data = table2array(dataTable);

% Center the data
avg = compute_average(data);
centered_data = data - avg;

%plotting the data for better visulisation
figure;
hold on;
scatter(data(:,1), data(:,2), 20, "red", "filled",'MarkerFaceAlpha',0.6);
scatter(centered_data(:,1), centered_data(:,2), 20, "blue", "filled",'MarkerFaceAlpha',0.6);
plot(mean(data(:,1)), mean(data(:,2)), "ks", "MarkerSize",10, "MarkerFaceColor","r"); % original mean
plot(mean(centered_data(:,1)), mean(centered_data(:,2)), "gs", "MarkerSize",10, "MarkerFaceColor","b"); % centered mean
title("Data visualization (Original vs Centered)");
xlabel("X-values");
ylabel("Y-values");
legend("Original Data", "Centered Data",'mean of data','mean of centered data');
grid on;
hold off;

%computing covariance for the centered data
cov_mat=compute_covariance(centered_data);

%eigen decomposition of covariance matrix
[eig_vec, eig_val_mat]=eig(cov_mat);
[eig_vals_sorted, indices]=sort(diag(eig_val_mat), 'descend');
eig_vec_sorted=eig_vec(:, indices);

%ratio of explained variance 
evr=eig_vals_sorted/sum(eig_vals_sorted);
disp('Ratio by each principal component:');
disp(evr);

%Projecting the data over the top 2 principal component
projected_data=centered_data*eig_vec_sorted(:,1:2);
figure;
scatter(projected_data(:,1), projected_data(:,2), 20, "filled");
title("Projection of data onto first 2 Principal Components");
xlabel("PC1");
ylabel("PC2");
grid on;

%% (b) Kernel PCA
%defining kernels
kernels = {'linear', 'polynomial', 'rbf'};
poly_degree = 3; % Degree for polynomial kernel
rbf_sigmas = [0.1, 1, 5]; % Various sigma values for RBF kernel

%Computing the K matrix
K=data'*data;

%Eigen decomposition and sorting them in descending order of eigen values
[e_vec,e_val_mat]=eig(K);
[e_val_sorted,indices]=sort(diag(e_val_mat),'descend');
e_vec_sorted=e_vec(:,indices);

%computing alpha_k
alpha_k=zeros(size(e_vec_sorted));
for k=1:size(e_val_sorted,2)
    if e_val_sorted>0
        alpha_k(:,k)=e_vec_sorted(:,k)/sqrt(size(centered_data,1)*e_val_sorted(k));
    else
        alpha_k(:,k)=0;
    end
end

%now the representations for the data will be the product between centered data and alpha_k
w=centered_data*alpha_k;