clc; close all;

% Anis diff parametrs
DELTA_T = 0.25;
K = 0.05;
T = 13;

% Rayleigh distr parametr
SCALE = 0.2707;

% Paths
NOISED_PATH = "./noised/";
FILTERED_PATH = "./filtered/";
ORIGINAL_PATH = "./original/";
GRAY_PATH = "./gray_images/";

% list of images in folder
FILES = dir("./original/*.jpg");

ssim_sum = 0;

for i = 1:length(FILES)
    img_name = FILES(i).name;

    % Open image and conver to grayscale
    original_img = mat2gray(rgb2gray(imread(ORIGINAL_PATH + img_name)));
    
    % Add noise to image
    noised_img = add_noise(original_img, SCALE);
    
    % Apply anis diff
    filtered_img = AnisotropicDiffusionExp(noised_img, T, K, DELTA_T);
     
    %figure('Name','Original'), imshow(original_img);
    %figure('Name','Filtered'), imshow(filtered_img);
    %figure('Name','Noised'), imshow(noised_img);
    
    % Check SSIM
    ssim_value = ssim(original_img, filtered_img);    
    data_log = [img_name, "ssim: ", (ssim_value)];
    ssim_sum = ssim_sum + ssim_value;
    disp(data_log);
    
    % Save images
    imwrite(noised_img, NOISED_PATH + img_name);
    imwrite(filtered_img, FILTERED_PATH + img_name);
    imwrite(original_img, GRAY_PATH + img_name);
end

% Average ssim
ssim_avg = ["ssim avg: ", ssim_sum / length(FILES)];
disp(ssim_avg)
