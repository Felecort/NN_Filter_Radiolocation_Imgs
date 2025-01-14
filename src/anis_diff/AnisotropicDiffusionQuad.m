function result_img = AnisotropicDiffusionQuad(noise_img, t, k, delta_t)
%%ïàðàìåòðû t, delta_t è k âûáèðàþòñÿ ýêñïåðèìåíòàëüíî
%%Ïàðàìåòð t — ýòî óðîâåíü ðàçìûòèÿ, íà êîòîðîì ìû õîòèì ïðåêðàòèòü ðàáîòó àëãîðèòìà, 
%%deltaT — øàã ïî âðåìåíè, k — ïàðàìåòð ôóíêöèè g
%%g_type - âèä ôóíêöèè g (1 èëè 2)

img = padarray(noise_img,[1,1]);
% img = add_copy_border(noise_img, [1,1]);
blur_img = zeros(size(noise_img));

level = 0;
while (level < t)
    N = img(1:end - 2,2:end - 1) - img(2:end-1,2:end-1);
    S = img(3:end,2:end - 1) - img(2:end-1,2:end-1);
    E = img(2:end - 1,3:end) - img(2:end-1,2:end-1);
    W = img(2:end - 1,1:end - 2) - img(2:end-1,2:end-1);
    

    blur_img = img(2:end-1,2:end-1) + ...
        + delta_t.*(g2(abs(N),k).*N + g2(abs(S),k).*S + ...
    	g2(abs(E),k).*E + g2(abs(W),k).*W);
    
    img(2:end-1,2:end-1) = blur_img;
    level = level + 1; % + delta_t;
end

result_img = blur_img;
result_img = max(min(result_img,1),0);
end

function g_result = g2(x,k)
    g_result = 1./(1 + (x./k).^2);
end