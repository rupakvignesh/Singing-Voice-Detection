function [] = plot_nn_accuracies(filename_train, filename_valid, filename_test)

xtr = textread(filename_train);
xvl = textread(filename_valid);
xts = textread(filename_test);
figure, plot(xtr,'o');
hold on;
plot(xvl,'o');
plot(xts,'o');
legend('Train','Valid','Test');
xlabel('Epochs');
ylabel('Accuracies');

end