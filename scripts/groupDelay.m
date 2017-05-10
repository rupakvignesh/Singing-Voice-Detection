%clear all; close all;
%% 1st pole
b = 1;
a1 = [1 -0.683 + .5i];
a2 = [1 -0.683 - .5i];
a = conv(a1,a2);
[h,n] = freqz(b,a,2001);
figure, subplot(3,1,1); zplane(b,a); title('System 1: Pole at a1')
subplot(3,1,2); plot(20*log(abs(h))); xlim([1 2001]);
xlabel('bin index'); ylabel('Log Magnitude Spectrum');
subplot(3,1,3); grpdelay(b,a);

%% 2nd pole
b_2 = 1;
a1_2 = [1 -0.2 + 0.8i];
a2_2 = [1 -0.2 - 0.8i];
a_2 = conv(a1_2,a2_2);
[h2,n2] = freqz(b_2,a_2,2001);
figure, subplot(3,1,1); zplane(b_2,a_2,2001); title('System 2: Pole at a2');
subplot(3,1,2); plot(20*log(abs(h2))); xlim([1 2001]);
xlabel('bin index'); ylabel('Log Magnitude Spectrum');
subplot(3,1,3); grpdelay(b_2,a_2);

%% Both poles
b_3 = 1;
a1_3 = conv(a1,a2);
a2_3 = conv(a1_2,a2_2);
a_3 = conv(a1_3,a2_3);
[h3,n3] = freqz(b_3,a_3,2001);
figure, subplot(3,1,1); zplane(b_3,a_3,2001); title('System 3: Poles at a1 and a2');
subplot(3,1,2); plot(20*log(abs(h3))); xlim([1 2001]);
xlabel('bin index'); ylabel('Log Magnitude Spectrum');
subplot(3,1,3); grpdelay(b_3,a_3);