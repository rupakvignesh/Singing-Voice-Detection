function mod_gd = derive_modified_gd(x)
X = fft(x);
Xr = real(X);
Xi = imag(X);
nx = (0:length(x)-1)'.*x;
Y = fft(nx);
Yr = real(Y);
Yi = imag(Y);
smoothened_spec = smooth(abs(X),5);
mod_gd = Xr.*Yr + Xi.*Yi;
mod_gd = mod_gd./smoothened_spec;
pos_scale = 1.5;
neg_scale = 0.2;
for i=1:length(x)
    if (mod_gd(i)>=0)
        mod_gd(i) = power(mod_gd(i),pos_scale);
    else
        mod_gd(i) = -1*power(abs(mod_gd(i)),neg_scale);
    end
end

mod_gd = mod_gd(1:1+length(mod_gd)/2)';

end