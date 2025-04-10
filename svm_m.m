% 1. Load all labels and prepare class names
labels_all = readtable("C:\Users\User\Desktop\dog_dataset\labels.csv");
disp(size(labels_all));
head(labels_all);

% Filter classes you need (same as your CLASS_NAMES)
CLASS_NAMES = {
    'boston_bull', 'dingo', 'pekinese', 'bluetick', 'golden_retriever', 'bedlington_terrier',
    'borzoi', 'basenji', 'scottish_deerhound', 'shetland_sheepdog', 'walker_hound',
    'maltese_dog', 'norfolk_terrier', 'african_hunting_dog', 'wire-haired_fox_terrier',
    'redbone', 'lakeland_terrier', 'boxer', 'doberman', 'otterhound', 'standard_schnauzer',
    'irish_water_spaniel', 'black-and-tan_coonhound', 'cairn', 'affenpinscher',
    'labrador_retriever', 'ibizan_hound', 'english_setter', 'weimaraner', 'giant_schnauzer',
    'groenendael', 'dhole', 'toy_poodle', 'border_terrier', 'tibetan_terrier',
    'norwegian_elkhound', 'shih-tzu', 'irish_terrier', 'kuvasz', 'german_shepherd',
    'greater_swiss_mountain_dog', 'basset', 'australian_terrier', 'schipperke',
    'rhodesian_ridgeback', 'irish_setter', 'appenzeller', 'bloodhound', 'samoyed',
    'miniature_schnauzer', 'brittany_spaniel', 'kelpie', 'papillon', 'border_collie',
    'entlebucher', 'collie', 'malamute', 'welsh_springer_spaniel', 'chihuahua', 'saluki',
    'pug', 'malinois', 'komondor', 'airedale', 'leonberg', 'mexican_hairless',
    'bull_mastiff', 'bernese_mountain_dog', 'american_staffordshire_terrier', 'lhasa',
    'cardigan', 'italian_greyhound', 'clumber', 'scotch_terrier', 'afghan_hound',
    'old_english_sheepdog', 'saint_bernard', 'miniature_pinscher', 'eskimo_dog',
    'irish_wolfhound', 'brabancon_griffon', 'toy_terrier', 'chow', 'flat-coated_retriever',
    'norwich_terrier', 'soft-coated_wheaten_terrier', 'staffordshire_bullterrier',
    'english_foxhound', 'gordon_setter', 'siberian_husky', 'newfoundland', 'briard',
    'chesapeake_bay_retriever', 'dandie_dinmont', 'great_pyrenees', 'beagle', 'vizsla',
    'west_highland_white_terrier', 'kerry_blue_terrier', 'whippet', 'sealyham_terrier',
    'standard_poodle', 'keeshond', 'japanese_spaniel', 'miniature_poodle', 'pomeranian',
    'curly-coated_retriever', 'yorkshire_terrier', 'pembroke', 'great_dane',
    'blenheim_spaniel', 'silky_terrier', 'sussex_spaniel', 'german_short-haired_pointer',
    'french_bulldog', 'bouvier_des_flandres', 'tibetan_mastiff', 'english_springer',
    'cocker_spaniel', 'rottweiler'
}
labels_filtered = labels_all(ismember(labels_all.breed, CLASS_NAMES), :);

% Split labels into train and test
split_ratio = 0.8; % 80% for training, 20% for testing
num_images = height(labels_filtered);
rand_indices = randperm(num_images);
train_count = round(split_ratio * num_images);

train_indices = rand_indices(1:train_count);
test_indices = rand_indices(train_count+1:end);

train_labels = labels_filtered(train_indices, :);
test_labels = labels_filtered(test_indices, :);

disp(['Number of training images: ', num2str(height(train_labels))]);
disp(['Number of testing images: ', num2str(height(test_labels))]);

% 2. Load a pre-trained CNN (e.g., AlexNet or ResNet)
net = alexnet(); % Use AlexNet
input_size = net.Layers(1).InputSize;

% 3. Feature extraction for training data
num_train_images = height(train_labels);
X_train_features = zeros(num_train_images, 4096); % Assuming AlexNet's 'fc7' layer outputs 4096 features
Y_train_data = zeros(num_train_images, 1);

for i = 1:num_train_images
    % Load and preprocess the image
    img_path = fullfile("C:\Users\User\Desktop\dog_dataset\train", [train_labels.id{i}, '.jpg']);
    img = imread(img_path);
    if size(img, 3) == 1
        img = repmat(img, [1, 1, 3]); % Convert grayscale to RGB
    end
    img = imresize(img, input_size(1:2)); % Resize to input size of CNN
    features = activations(net, img, 'fc7', 'OutputAs', 'rows'); % Extract features from 'fc7' layer
    X_train_features(i, :) = features;
    Y_train_data(i) = find(strcmp(CLASS_NAMES, train_labels.breed{i})); % Encode labels
end

Y_train_data = categorical(Y_train_data, 1:numel(CLASS_NAMES), CLASS_NAMES);

% 4. Feature extraction for testing data
num_test_images = height(test_labels);
X_test_features = zeros(num_test_images, 4096); % Same as training
Y_test_data = zeros(num_test_images, 1);

for i = 1:num_test_images
    % Load and preprocess the image
    img_path = fullfile("C:\Users\User\Desktop\dog_dataset\test_aiml", [test_labels.id{i}, '.jpg']); % Assuming test images are in the train folder
    img = imread(img_path);
    if size(img, 3) == 1
        img = repmat(img, [1, 1, 3]); % Convert grayscale to RGB
    end
    img = imresize(img, input_size(1:2)); % Resize to input size of CNN
    features = activations(net, img, 'fc7', 'OutputAs', 'rows'); % Extract features
    X_test_features(i, :) = features;
    Y_test_data(i) = find(strcmp(CLASS_NAMES, test_labels.breed{i})); % Encode labels
end

Y_test_data = categorical(Y_test_data, 1:numel(CLASS_NAMES), CLASS_NAMES);

disp(['Train features size: ', num2str(size(X_train_features))]);
disp(['Test features size: ', num2str(size(X_test_features))]);

% 5. Train an SVM model with progress monitoring
cv = cvpartition(num_train_images, 'HoldOut', 0.2); % 80% for training, 20% for validation

% Placeholder for accuracy
train_accuracies = zeros(cv.NumTestSets, 1);

% For each training fold, train the SVM and track progress
for i = 1:cv.NumTestSets
    train_indices = cv.training(i); % Get training indices for current fold
    test_indices = cv.test(i); % Get validation indices for current fold

    % Train the SVM using the current fold's data
    svm_model = fitcecoc(X_train_features(train_indices, :), Y_train_data(train_indices), 'Coding', 'onevsall', 'Verbose', 2);
    
    % Predict using the validation set
    Y_pred = predict(svm_model, X_train_features(test_indices, :));
    
    % Calculate accuracy for this fold
    train_accuracies(i) = sum(Y_pred == Y_train_data(test_indices)) / numel(Y_pred);
    
    % Plot the training progress
    figure(1);
    plot(1:i, train_accuracies(1:i), 'b', 'LineWidth', 2);
    xlabel('Fold');
    ylabel('Accuracy');
    title('SVM Training Progress');
    drawnow;
end

disp('SVM Model Trained.');

% 6. Evaluate the SVM model on the full test set
Y_pred = predict(svm_model, X_test_features);
accuracy = sum(Y_pred == Y_test_data) / numel(Y_test_data);
fprintf('Test Accuracy: %.2f%%\n', accuracy * 100);

% 7. Save the trained SVM model
save('dog_svm_classifier.mat', 'svm_model');
disp('SVM model saved as dog_svm_classifier.mat.');