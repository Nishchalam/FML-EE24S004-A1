%% (1) You are given a dataset with 1000 data points each in R2 . 
%% (a) Write a piece of code to run the PCA algorithm on this data-set. How much of  the variance in the data-set is explained by each of the principal components?
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
%these eigen vectors are the vectors which are in line with most variance
%in the data
eig_vec_sorted=eig_vec(:, indices);

%ratio of explained variance 
evr=eig_vals_sorted/sum(eig_vals_sorted);
disp('Ratio by each principal component:');
disp(evr);

%Plotting centered data with PCs overlay
figure;
hold on;
scatter(centered_data(:,1),centered_data(:,2), 20,"green","filled",'MarkerFaceAlpha',0.6);
scale=2; % scaling factor to make arrows visible
quiver(0,0,scale*eig_vec_sorted(1,1),scale*eig_vec_sorted(2,1),'r','LineWidth',2,'MaxHeadSize',2);
quiver(0,0,scale*eig_vec_sorted(1,2),scale*eig_vec_sorted(2,2),'b','LineWidth',2,'MaxHeadSize',2);
title("Centered Data with Principal Component Directions");
xlabel("X-values");
ylabel("Y-values");
legend("Centered Data","PC1","PC2");
grid on;
axis equal;
hold off;

%Explained Variance Ratio Bar Plot
figure;
bar(evr,'FaceColor',[0.2 0.4 0.8]);
title("Explained variance (fraction) per component");
xlabel("Principal Component");
ylabel("Explained Variance Ratio");
grid on;

% Project data into PC space
projected_data=centered_data*eig_vec_sorted;

% Projection of data into PC space (PC1 vs PC2)
figure;
scatter(projected_data(:,1), projected_data(:,2), 20, 'filled');
title("Projection of Centered Data onto Principal Components");
xlabel("PC1");
ylabel("PC2");
grid on;


% Reconstruct from PCs (using all components)
reconstructed_data=projected_data*eig_vec_sorted';

% Compute reconstruction error (Mean Squared Error)
errors=centered_data-reconstructed_data;
mse=mean(sum(errors.^2,2));

% Plot comparison
figure;
hold on;
scatter(centered_data(:,1),centered_data(:,2),20,'b','filled','MarkerFaceAlpha',0.5);
scatter(reconstructed_data(:,1),reconstructed_data(:,2),20,'r','filled','MarkerFaceAlpha',0.5);
title("Centered Data vs PCA Reconstruction");
xlabel("X-values");
ylabel("Y-values");
legend("Original Centered Data","Reconstructed from PCs");
grid on;
axis equal;

% Add error text on the plot
dim = [.15 .8 .1 .1]; % position of annotation (normalized figure units)
annotation('textbox',dim,'String',sprintf('Reconstruction MSE = %.4f',mse),'FitBoxToText','on','BackgroundColor','w');
hold off;

pc1_vector = eig_vec_sorted(:,1);         % First principal component
projected_pc1 = centered_data * pc1_vector; % Projection onto PC1 (1D)
reconstructed_pc1 = projected_pc1 * pc1_vector'; % Back to 2D

% Compute reconstruction error (MSE) for PC1 only
errors_pc1 = centered_data - reconstructed_pc1;
mse_pc1 = mean(sum(errors_pc1.^2,2));

% Plot comparison: original vs reconstruction using only PC1
figure;
hold on;
scatter(centered_data(:,1), centered_data(:,2), 20, 'b','filled','MarkerFaceAlpha',0.5);
scatter(reconstructed_pc1(:,1), reconstructed_pc1(:,2), 20, 'r','filled','MarkerFaceAlpha',0.5);
title("Centered Data vs PCA Reconstruction (Only PC1)");
xlabel("X-values"); ylabel("Y-values");
legend("Original Centered Data","Reconstruction using PC1");
grid on; axis equal;

% Show error as text
dim = [.15 .8 .1 .1];
annotation('textbox',dim,'String',sprintf('MSE (PC1 only) = %.4f',mse_pc1),...
           'FitBoxToText','on','BackgroundColor','w');
hold off;


%% (b) Write a piece of code to implement the Kernel PCA algorithm on this dataset.Explore various kernels discussed in class.For each Kernel, plot the projection of each point in the dataset onto the top-2 principal components. Use one plot for each kernel - In case of RBF kernel, use a different plot for each value of σ that you use.
kernels={'linear', 'polynomial', 'rbf'};
poly_degree=3; % Degree for polynomial kernel
rbf_sigmas=[0.1, 1, 5]; % Different sigma values for RBF kernel

% loop over kernels
for kidx=1:length(kernels)
    kernel_type=kernels{kidx};
    if strcmp(kernel_type,'linear')
        K=centered_data*centered_data'; % linear kernel
        figure_title='Kernel PCA with Linear Kernel';
        sigmas=1; % dummy
    elseif strcmp(kernel_type,'polynomial')
        K=(centered_data*centered_data' + 1).^poly_degree; % polynomial kernel
        figure_title=sprintf('Kernel PCA with Polynomial Kernel (deg=%d)', poly_degree);
        sigmas=1; % dummy
    elseif strcmp(kernel_type,'rbf')
        sigmas=rbf_sigmas; % loop over RBF sigmas
    end
    
    % if RBF, loop over sigma values
    for s=1:length(sigmas)
        if strcmp(kernel_type,'rbf')
            sigma=sigmas(s);
            sq_dists=pdist2(centered_data,centered_data,'euclidean').^2;
            K=exp(-sq_dists/(2*sigma^2));
            figure_title=sprintf('Kernel PCA with RBF Kernel (σ=%.2f)', sigma);
        end

        % Eigen decomposition of K
        [e_vec,e_val_mat]=eig(K);
        [e_val_sorted,indices]=sort(diag(e_val_mat),'descend');
        e_vec_sorted=e_vec(:,indices);

        % Normalize eigenvectors (alphas)
        alpha_k=zeros(size(e_vec_sorted));
        for k=1:length(e_val_sorted)
            if e_val_sorted(k) > 1e-8
                alpha_k(:,k)=e_vec_sorted(:,k)/sqrt(e_val_sorted(k));
            end
        end

        % Representations of the data in feature space
        projected_kpca=K*alpha_k(:,1:2);

        % Plotting
        figure;
        scatter(projected_kpca(:,1),projected_kpca(:,2),20,'filled');
        title(figure_title);
        xlabel('PC1'); ylabel('PC2');
        grid on;
    end
end

%% (c) Which Kernel do you think is best suited for this dataset and why?