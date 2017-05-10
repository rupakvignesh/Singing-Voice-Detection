function D = modgd(x, f, w, h, sr)

if nargin < 2;  f = 256; end
if nargin < 3;  w = f; end
if nargin < 4;  h = 0; end
if nargin < 5;  sr = 8000; end

% expect x as a row
if size(x,1) > 1
  x = x';
end
 
s = length(x);

if length(w) == 1
  if w == 0
    % special case: rectangular window
    win = ones(1,f);
  else
    if rem(w, 2) == 0   % force window to be odd-len
      w = w + 1;
    end
    halflen = (w-1)/2;
    halff = f/2;   % midpoint of win
    halfwin = 0.5 * ( 1 + cos( pi * (0:halflen)/halflen));
    win = zeros(1, f);
    acthalflen = min(halff, halflen);
    win((halff+1):(halff+acthalflen)) = halfwin(1:acthalflen);
    win((halff+1):-1:(halff-acthalflen+2)) = halfwin(1:acthalflen);
  end
else
  win = w;
end

w = length(win);
% now can set default hop
if h == 0
  h = floor(w/2);
end

c = 1;

% pre-allocate output array
d = zeros((1+f/2),1+fix((s-f)/h));

for b = 0:h:(s-f)
  u = win.*x((b+1):(b+f));
  t = fft(u);
  Mag_spec = abs(fft(u));

  X_R = real(fft(u)); X_I = imag(fft(u));
  n_X = zeros(1,length(u));
 	for index2 = 2:1:length(u)
 	n_X(index2) = (index2-1)*u(index2);
 	end
 Y_R = real(fft(n_X)); Y_I = imag(fft(n_X));

 param1 = 5;
 smoothened_Mag_spec = smooth(Mag_spec,param1);
 Mod_GD = X_R.*Y_R + X_I.*Y_I;
 Mod_GD = (Mod_GD)./smoothened_Mag_spec'; 

gdPosscale = 1.5;
gdNegscale = 0.2;
 index2 = 1;
   while (index2 < length(Mod_GD))
             if (Mod_GD(index2) > 0 )
                    Mod_GD(index2) = power(Mod_GD(index2),gdPosscale);
             else
             abs_val = abs(Mod_GD(index2));
               Mod_GD(index2) = -1*power(abs_val,gdNegscale);
             end
                    index2 = index2 + 1;
   end
  d(:,c) = Mod_GD(1:(1+f/2))';
  c = c+1;
end;

% If no output arguments, plot a modgdgram
if nargout == 0
  tt = [0:size(d,2)]*h/sr;
  ff = [0:size(d,1)]*sr/f;
  imagesc(tt,ff,(abs(d)));
  axis('xy');
  xlabel('time / sec');
  ylabel('freq / Hz')
  % leave output variable D undefined
else
  % Otherwise, no plot, but return modgd
  D = d;
end

