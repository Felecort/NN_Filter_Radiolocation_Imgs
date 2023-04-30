function noised_img = add_noise(img, scale)
%%DESCRIPTION

[rows, columns, ~] = size(img);
% Gen noise
noise = raylrnd(scale, rows, columns);

% Apply noise to image
noised_img = img + img.*noise;

% transform to -> [0, 1] range
noised_img = max(min(noised_img, 1), 0);

end