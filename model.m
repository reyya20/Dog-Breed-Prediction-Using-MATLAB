% 1. Load all labels and prepare class names
labels_all = readtable('C:\Users\User\Desktop\dog_dataset\labels.csv');
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

% 2. Preprocess training images
image_size = [224, 224, 3];
num_train_images = height(train_labels);
X_train_data = zeros([image_size, num_train_images], 'single');
Y_train_data = zeros(num_train_images, 1);

for i = 1:num_train_images
    % Load and preprocess the image
    img_path = fullfile('C:\Users\User\Desktop\dog_dataset\train', [train_labels.id{i}, '.jpg']);
    img = imread(img_path);
    if size(img, 3) == 1
        img = repmat(img, [1, 1, 3]); % Convert grayscale to RGB
    end
    img = imresize(img, [224, 224]); % Resize image
    X_train_data(:,:,:,i) = single(img) / 255.0; % Normalize to [0, 1]
    Y_train_data(i) = find(strcmp(CLASS_NAMES, train_labels.breed{i})); % Encode labels
end

Y_train_data = categorical(Y_train_data, 1:numel(CLASS_NAMES), CLASS_NAMES);

% 3. Preprocess testing images
num_test_images = height(test_labels);
X_test_data = zeros([image_size, num_test_images], 'single');
Y_test_data = zeros(num_test_images, 1);

for i = 1:num_test_images
    % Load and preprocess the image
    img_path = fullfile('C:\Users\User\Desktop\dog_dataset\train', [test_labels.id{i}, '.jpg']); % Assuming test images are in the train folder
    img = imread(img_path);
    if size(img, 3) == 1
        img = repmat(img, [1, 1, 3]); % Convert grayscale to RGB
    end
    img = imresize(img, [224, 224]); % Resize image
    X_test_data(:,:,:,i) = single(img) / 255.0; % Normalize to [0, 1]
    Y_test_data(i) = find(strcmp(CLASS_NAMES, test_labels.breed{i})); % Encode labels
end

Y_test_data = categorical(Y_test_data, 1:numel(CLASS_NAMES), CLASS_NAMES);

disp(['Train dataset size: ', num2str(size(X_train_data))]);
disp(['Test dataset size: ', num2str(size(X_test_data))]);

% 4. Define the CNN architecture
layers = [
    imageInputLayer([224 224 3], 'Normalization', 'none', 'Name', 'input')
    
    convolution2dLayer(5, 64, 'Padding', 'same', 'Name', 'conv1')
    reluLayer('Name', 'relu1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool1')
    
    convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv2')
    reluLayer('Name', 'relu2')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool2')
    
    flattenLayer('Name', 'flatten')
    
    fullyConnectedLayer(128, 'Name', 'fc1')
    reluLayer('Name', 'relu3')
    
    fullyConnectedLayer(numel(CLASS_NAMES), 'Name', 'fc2')
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];

% 5. Set training options
options = trainingOptions('adam', ...
    'MaxEpochs', 20, ...
    'MiniBatchSize', 32, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', true, ...
    'Plots', 'training-progress');

% 6. Train the model
net = trainNetwork(X_train_data, Y_train_data, layers, options);

% 7. Evaluate on test data
YPred = classify(net, X_test_data);
accuracy = sum(YPred == Y_test_data) / numel(Y_test_data);
fprintf('Test Accuracy: %.2f%%\n', accuracy * 100);

% 8. Display a test image and prediction
figure;
imshow(X_test_data(:,:,:,1)); % Show the first test image
title(['True: ', char(Y_test_data(1)), ', Predicted: ', char(YPred(1))]);

% 9. Save the trained model
save('dog_classifier.mat', 'net');
disp('Model saved as dog_classifier.mat');
