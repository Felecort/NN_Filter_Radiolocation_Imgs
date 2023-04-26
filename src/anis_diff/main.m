clear all; clc;

t = 13;
k = 0.05;
delta_t = 0.25;

source_path = "./noised/";
out_path = "./filtered/";
files = dir("./noised/*.jpg");

for i = 1:length(files)
    disp(i)
    name = files(i).name;
    img = mat2gray((imread(source_path + name)));
    

    
    filtered_image = AnisotropicDiffusionExp(img, t, k, delta_t);
    %imshow(filtered_image);
    imwrite(filtered_image, out_path + name);
end
