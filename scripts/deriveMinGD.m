function minGD = deriveMinGD(x)

X = fft(x);
gamma = 0.001
X_2 = abs(X).^(2*gamma); %power spectrum

%%
xc = ifft(log(X_2));
hann = hanning(length(xc));
xc_min = hann(length(hann)/2+1:end).*xc(1:length(xc)/2);
xc_min(round(length(xc_min)/2):end) = 0;

Xmin = fft(xc_min,2*length(xc_min));
Xphase = unwrap(angle(Xmin));
Xgd = [-diff(Xphase);0];

minGD = Xgd(1:length(Xgd)/2 +1);


end