

import matplotlib.pyplot as plt


weights = [2.5, 3.0, 4.5, 5.0, 6.5, 7.0, 8.0, 9.5]
heights = [20, 22, 25, 27, 40, 42, 45, 48]
labels  = ['cat','cat','cat','cat','dog','dog','dog','dog']

for w, h, l in zip(weights, heights, labels):
    print(w, 'kg,', h, 'cm ->', l)


def rule_based_classifier(weight, height):
    if weight < 5 and height < 30:
        return 'cat'
    else:
        return 'dog'

print(rule_based_classifier(4.5, 25))
print(rule_based_classifier(7.0, 42))
cat_weight_avg = (2.5 + 3.0 + 4.5 + 5.0) / 4
cat_height_avg = (20 + 22 + 25 + 27) / 4
dog_weight_avg = (6.5 + 7.0 + 8.0 + 9.5) / 4
dog_height_avg = (40 + 42 + 45 + 48) / 4

print('Average cat:', cat_weight_avg, 'kg,', cat_height_avg, 'cm')
print('Average dog:', dog_weight_avg, 'kg,', dog_height_avg, 'cm')


def ai_classifier(weight, height):
    distance_to_cat = abs(weight - cat_weight_avg) + abs(height - cat_height_avg)
    distance_to_dog = abs(weight - dog_weight_avg) + abs(height - dog_height_avg)
    if distance_to_cat < distance_to_dog:
        return 'cat'
    else:
        return 'dog'

new_animals = [(5.2, 29), (6.8, 33), (4.8, 31)]

for w, h in new_animals:
    rule_pred = rule_based_classifier(w, h)
    ai_pred = ai_classifier(w, h)
    print(f'{w} kg, {h} cm -> rule says {rule_pred}, AI says {ai_pred}')



plt.figure(figsize=(6, 5))
colors = ['orange' if l == 'cat' else 'steelblue' for l in labels]
plt.scatter(weights, heights, c=colors, s=80, label='training data')

for w, h in new_animals:
    plt.scatter(w, h, c='red', marker='x', s=100)

plt.xlabel('Weight (kg)')
plt.ylabel('Height (cm)')
plt.title('Cat vs Dog: training data and new animals (red x)')
plt.show()