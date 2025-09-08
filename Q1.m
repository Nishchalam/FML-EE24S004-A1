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
%User defined for computing covariance of 
%% Main program
dataTable = readtable("dataset1-assignment1 - Sheet1.csv");
data = table2array(dataTable);

% Center the data
avg = compute_average(data);
centered_data = data - avg;
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

