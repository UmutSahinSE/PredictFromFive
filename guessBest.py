import csv
from itertools import combinations
from prettytable import PrettyTable
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import matplotlib.pyplot as plt

def calculateRegCoef(arr1,arr2):

    arr1 = [int(x) for x in arr1]
    arr2 = [int(x) for x in arr2]
    arr1=np.array(arr1)
    arr2=np.array(arr2)
    avg1 = sum(arr1) / len(arr1)
    print("avg of arr1")
    print(avg1)
    avg2 = sum(arr2) / len(arr2)
    print("avg of arr2")
    print(avg2)
    s1 = []
    s2 = []
    for i in arr1:
        s1.append(i-avg1)
    print("avg-elem of l1")
    print(s1)
    for i in arr2:
        s2.append(i-avg2)
    print("avg-elem of l2")
    print(s2)
    a1 = [a*b for a,b in zip(s1,s2)]
    print(a1)
    ss1 = [a*b for a,b in zip(s2,s2)]
    print(ss1)
    b1 = sum(a1)/sum(ss1)
    print(sum(a1))
    print(sum(ss1))
    print(b1)
    c1 = avg1-b1*avg2
    print(c1)
    return [b1,c1]

def calculateRSS(arr, arr2):
    arr = [float(x) for x in arr]
    arr2 = [float(x) for x in arr2]
    arr = np.array(arr)
    arr2 = np.array(arr2)
    sub = [a-b for a,b in zip(arr,arr2)]
    sqr = [a*b for a, b in zip(sub, sub)]
    summ = 0
    for i in sqr:
        summ += i
    return summ

def calculate_R_square(predict, y_values):
    RSS = calculateRSS(predict,y_values)
    mean_of_y = sum(y_values)/len(y_values)
    pred_minus_mean = [x - mean_of_y for x in y_values]
    sqr = [a * b for a, b in zip(pred_minus_mean, pred_minus_mean)]
    starting_variance = sum(sqr)
    return 1-RSS/starting_variance

def calculateAdjustedR2(R2, number_of_parameters, number_of_independent):
    top = (1-R2)*(number_of_independent-1)
    bottom = number_of_independent-number_of_parameters-1
    return 1-(top/bottom)

def fill_input_matrix(knot_values,x_values):
    list_to_be_filled = []
    for n in x_values:
        row_of_X = []
        for pw in range(4):
            row_of_X.append(pow(n,pw))
        for k in knot_values:
            a = pow(n - k, 3)
            if a < 0:
                a=0
            row_of_X.append(a)
        list_to_be_filled.append(row_of_X)
    return np.array(list_to_be_filled)

def calculate_coef(input_matrix,y_values):
    Xtran = np.transpose(input_matrix)
    mult = np.matmul(Xtran, input_matrix)
    inv = np.linalg.inv(mult)
    final_mult = np.matmul(inv, Xtran)
    return np.matmul(final_mult,y_values)

def predict_new(reg_for_preknot,reg_for_postknot,x_values,pre_x_indexes,post_x_indexes):
    predicted_y_values = []

    for x in x_values:
        if x[1]>32:
            picked_x = []
            for i in post_x_indexes:
                picked_x.append(x[i])
            predicted_y_values.append(reg_for_postknot.predict([picked_x]))
        else:
            picked_x = []
            for i in pre_x_indexes:
                picked_x.append(x[i])
            predicted_y_values.append(reg_for_preknot.predict([picked_x]))
    return predicted_y_values

list_known_x_values = []
list_y_values = []
list_x_values_without_y = []

with open('./data.csv', encoding='latin-1') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        if len(row[0]) > 4:
            continue
        elif row[-1]:
            list_known_x_values.append(row[1:-1])
            list_y_values.append(row[-1])
        else:
            list_x_values_without_y.append(row[1:-1])

int_known_x_values = []
int_x_values_without_y = []
int_y_values = [int(x) for x in list_y_values]
for l in list_known_x_values:
    int_known_x_values.append([int(x) for x in l])
for l in list_x_values_without_y:
    int_x_values_without_y.append([int(x) for x in l])

table = PrettyTable(["X1","X2","X3","X4","X5"])
for x in int_x_values_without_y:
    table.add_row([x[0],x[1],x[2],x[3],x[4]])
print(table)

known_x_values = np.array(int_known_x_values)
y_values = np.array(int_y_values)
x_values_without_y = np.array(int_x_values_without_y)

best_param_comb_of_each = [None,None,None,None,None]
least_RSS_of_each = [99999999999999999999,99999999999999999999,99999999999999999999,99999999999999999999,99999999999999999999]
predict_of_best = [None,None,None,None,None]
coef_of_best = [None,None,None,None,None]

for number_of_parameters in range(1,6):
    index_combinations = list(combinations(range(5),number_of_parameters))
    index_combinations = [list(i) for i in index_combinations]
    for combination in index_combinations:
        picked_known_x_values_for_calc = []
        for x in known_x_values:
            temp = []
            for c in combination:
                temp.append(x[c])
            picked_known_x_values_for_calc.append(temp)
        picked_x_values_without_y_for_calc = []
        # for x in x_values_without_y:
        #     temp = []
        #     for c in combination:
        #         temp.append(x[c])
        #     picked_x_values_without_y_for_calc.append(temp)

        reg = LinearRegression()
        reg.fit(picked_known_x_values_for_calc,y_values)
        predict = reg.predict(picked_known_x_values_for_calc)

        #plt.scatter(range(len(y_values)),y_values)
        # plt.scatter(x1, predict)
        #plt.show()

        RSS = calculateRSS(predict,y_values)
        if RSS < least_RSS_of_each[number_of_parameters-1]:
            least_RSS_of_each[number_of_parameters - 1] = RSS
            best_param_comb_of_each[number_of_parameters-1] = combination
            predict_of_best[number_of_parameters-1] = predict
            coef_of_best[number_of_parameters - 1] = reg.coef_

table = PrettyTable(["Combination","RSS","Adjusted R^2","Coefficients"])

for p in range(5):
    # xx = []
    # for i in known_x_values:
    #     xx.append(i[2])
    # Z = [x for _, x in sorted(zip(xx, y_values))]
    # plt.scatter(sorted(xx), Z)
    # Z = [x for _, x in sorted(zip(xx, predict_of_best[p]))]
    # plt.scatter(sorted(xx), Z)
    # plt.show()
    R2 = calculate_R_square(predict_of_best[p], y_values)
    adjR2 = calculateAdjustedR2(R2, p+1, 100)
    combination_string = "["
    for i in best_param_comb_of_each[p]:
        combination_string += str(i) + ","
    table.add_row([combination_string[:-1]+"]",str(least_RSS_of_each[p]),str(adjR2),coef_of_best[p]])
table.align["RSS"] = "r"
table.align["Adjusted R^2"] = "l"
table.align["Coefficients"] = "l"
print(table)

least_RSS_of_ridges = [99999999999999999999,99999999999999999999,99999999999999999999,99999999999999999999,99999999999999999999]
predict_of_ridges = [None,None,None,None,None]
coef_of_ridges = [None,None,None,None,None]
penalties = [None,None,None,None,None]
least_penalty_gaps = [10,10,10,10,10]

print("Applying Ridge Regression")
table = PrettyTable(["Combination","Tuner","RSS","Adjusted R^2","Coefficients"])
for combination in best_param_comb_of_each:
    if len(combination) is 1:
        continue
    picked_known_x_values_for_calc = []
    for x in known_x_values:
        temp = []
        for c in combination:
            temp.append(x[c])
        picked_known_x_values_for_calc.append(temp)
    for j in range(0,10):
        penalty = j/10
        ridgereg = Ridge(penalty,normalize=True)
        ridgereg.fit(picked_known_x_values_for_calc,y_values)
        highest_two_params = sorted(ridgereg.coef_)
        predict = ridgereg.predict(picked_known_x_values_for_calc)
        RSS = calculateRSS(predict, y_values)
        R2 = calculate_R_square(predict, y_values)
        adjR2 = calculateAdjustedR2(R2, len(combination), 100)
        combination_string = "["
        for i in combination:
            combination_string += str(i) + ","
        table.add_row([combination_string[:-1] + "]", str(penalty), str(RSS), str(adjR2),
                       str(ridgereg.coef_)])
    table.add_row(["","","","",""])

table.align["RSS"] = "l"
table.align["Adjusted R^2"] = "l"
table.align["Coefficients"] = "l"
print(table)

print("Applying Lasso")

table = PrettyTable(["Combination","Tuner","RSS","Adjusted R^2","Coefficients"])
for combination in best_param_comb_of_each:
    if len(combination) is 1:
        continue
    picked_known_x_values_for_calc = []
    for x in known_x_values:
        temp = []
        for c in combination:
            temp.append(x[c])
        picked_known_x_values_for_calc.append(temp)
    for j in range(0,10):
        penalty = j/10
        lassoreg = Lasso(alpha=penalty,normalize=True)
        lassoreg.fit(picked_known_x_values_for_calc,y_values)
        highest_two_params = sorted(lassoreg.coef_)
        predict = lassoreg.predict(picked_known_x_values_for_calc)
        RSS = calculateRSS(predict, y_values)
        R2 = calculate_R_square(predict, y_values)
        adjR2 = calculateAdjustedR2(R2, len(combination), 100)
        combination_string = "["
        for i in combination:
            combination_string += str(i) + ","
        table.add_row([combination_string[:-1] + "]", str(penalty), str(RSS), str(adjR2),
                       str(lassoreg.coef_)])
    table.add_row(["","","","",""])

table.align["RSS"] = "l"
table.align["Adjusted R^2"] = "l"
table.align["Coefficients"] = "l"
print(table)

only_x2 = []
for i in known_x_values:
        only_x2.append(i[1])

Z = [x for _,x in sorted(zip(only_x2,y_values))]
plt.scatter(sorted(only_x2),Z)
plt.show()

only_x3_with_cond = []
only_x2_with_cond = []
only_x5_lower = []
only_x5_higher = []
only_x4_higher = []

y_3 = []
y_2 = []
for i in range(len(known_x_values)):
    if known_x_values[i][1]<32:
        only_x2_with_cond.append(known_x_values[i][2])
        only_x5_lower.append(known_x_values[i][4])
        y_2.append(y_values[i])
    else:
        only_x3_with_cond.append(known_x_values[i][1])
        only_x4_higher.append(known_x_values[i][3])
        y_3.append(y_values[i])

Z = [x for _,x in sorted(zip(only_x2_with_cond,y_2))]
plt.scatter(sorted(only_x2_with_cond),Z)
plt.show()

Z = [x for _,x in sorted(zip(only_x3_with_cond,y_3))]
plt.scatter(sorted(only_x3_with_cond),Z)
plt.show()

preknot_reg1 = LinearRegression()
x_2_two_dim = []
for i in only_x2_with_cond:
    x_2_two_dim.append([i])
preknot_reg1.fit(x_2_two_dim,y_2)

postknot_reg1 = LinearRegression()
x_3_two_dim = []
for i in only_x3_with_cond:
    x_3_two_dim.append([i])
postknot_reg1.fit(x_3_two_dim,y_3)

predictions1 = predict_new(preknot_reg1,postknot_reg1,known_x_values,[2],[2])

RSS1 = calculateRSS(predictions1,y_values)
print("RSS without x5 involved: "+str(RSS1))
R2_1 = calculate_R_square(predictions1,y_values)
print("R^2 without x5 involved: "+str(R2_1))

preknot_reg2 = LinearRegression()
preknot_x_values = []
for i in range(len(only_x2_with_cond)):
    preknot_x_values.append([only_x2_with_cond[i],only_x5_lower[i]])
preknot_reg2.fit(preknot_x_values,y_2)


postknot_reg2 = LinearRegression()
postknot_x_values = []

for i in range(len(only_x3_with_cond)):
    postknot_x_values.append([only_x3_with_cond[i],only_x4_higher[i]])
postknot_reg2.fit(postknot_x_values,y_3)

print(preknot_reg2.coef_)
print(postknot_reg2.coef_)

predictions2 = predict_new(preknot_reg2,postknot_reg2,known_x_values,[2,4],[1,3])

RSS2 = calculateRSS(predictions2,y_values)
print("RSS of best: "+str(RSS2))
R2_2 = calculate_R_square(predictions2,y_values)
print("R^2 of best: "+str(R2_2))

predict_unknown = predict_new(preknot_reg2,postknot_reg2,x_values_without_y,[2,4],[1,3])
predict_unknown = [i[0] for i in predict_unknown]
for i in range(len(predict_unknown)):
    print(str(i+1)+". "+str(predict_unknown[i]))

outfile = open('./result.csv','w')
out = csv.writer(outfile)
out.writerows(map(lambda x: [x], predict_unknown))
outfile.close()


# cross_validation_error_each = []
#
#
# for i in range(10):
#
#     x_to_fit = np.concatenate((known_x_values[:i*10],known_x_values[(i+1)*10:]),axis=0)
#     y_to_fit = np.concatenate((y_values[:i*10],y_values[(i+1)*10:]),axis=0)
#     x_to_test = known_x_values[i*10:(i+1)*10]
#     y_to_test = y_values[i*10:(i+1)*10]
#
#     reg = LinearRegression()
#     reg.fit(x_to_fit, y_to_fit)
#     predict = reg.predict(x_to_test)
#     error_of_step = (calculateRSS(predict, y_to_test)/90)/10
#     cross_validation_error_each.append(error_of_step)
#
# CVE = sum(cross_validation_error_each)
# print("Cross validation error")
# print(CVE)
