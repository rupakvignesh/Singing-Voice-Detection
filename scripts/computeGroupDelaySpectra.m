function [groupDelaySpectra, logE] = computeGroupDelaySpectra(x,fs,winSize,hopSize,window)
% function [groupDelaySpectra, logE] = computeGroupDelaySpectra(x,fs,winSize,hopSize,window)
% x: audio
% fs: sampling rate in Hz
% winSize: window length in sec
% hopSize: frame shift in sec
% window: hann(0), hamming(1), rect(2)
N = length(x);
winSize = round(fs*winSize);
hopSize = round(fs*hopSize);
nCols = fix((N-winSize+hopSize)/(hopSize));
winSize = 2^(ceil(log(winSize)/log(2)));

% Pad zeros
temp = (length(x)-winSize)/hopSize;
n_zeros = round(hopSize*(ceil(temp)-temp));
x = [x; zeros(n_zeros,1)];
N = length(x); % new length of x

groupDelaySpectra = zeros(winSize/2 +1,nCols);
if window==0
    h = hanning(winSize);
elseif window ==1
    h = hamming(winSize);
else
    h = ones(winSize,1); 
end

pos = 1;
i = 1;
while(pos+winSize<=N)
    groupDelaySpectra(:,i) = abs((derive_modified_gd(x(pos:pos+winSize-1).*h)));
    pos = pos + hopSize;
    i = i+1;
end

logE = sum(log(abs(groupDelaySpectra)));

end