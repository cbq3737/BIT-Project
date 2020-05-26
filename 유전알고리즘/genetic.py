import tensorflow as tf
import numpy as np
import random
tf.set_random_seed(777)

model = tf.global_variables_initializer()
sess = tf.Session()
nb_classes = 5  # 1 ~ 3

X = tf.placeholder(tf.float32, shape = [None,1])
Y = tf.placeholder(tf.int32, shape = [None,1])
Y_one_hot = tf.one_hot(Y, nb_classes)  # one hot
Y_one_hot = tf.reshape(Y_one_hot, [-1, nb_classes])

W = tf.Variable(tf.random_normal([1, nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

saver = tf.train.Saver()
# save_path = "./pingpongsaved.cpkt"
save_path = "./softsaved.cpkt"

saver.restore(sess, save_path)

target_score = 5
max_servo=160
min_servo=95

####################선형회귀용해서 값받아오기
#소수점 제거
def digit(value):
    return int(value*100)/100


#선형회귀 사용(매개변수 없음)
def receive_val1():

    logits = tf.matmul(X, W) + b
    hypothesis = tf.nn.softmax(logits)

    # 조건 랜덤생성
    servo = random.randint(min_servo, max_servo)

    ###결과값 받기
    x_data = np.array([[servo / 10]])

    sess.run(model)
    prediction = tf.argmax(hypothesis, 1)
    dict = sess.run(prediction,feed_dict={X:x_data})

    data_list=[]
    data_list.append(digit(servo))
    data_list.append(digit(float(dict[0]+1)))

    return data_list

#선형회귀 사용(매개변수 있음)
def receive_val2(servo):

    logits = tf.matmul(X, W) + b
    hypothesis = tf.nn.softmax(logits)


    ###결과값 받기
    x_data = np.array([[servo / 10]])

    sess.run(model)
    prediction = tf.argmax(hypothesis, 1)
    dict = sess.run(prediction,feed_dict={X:x_data})

    data_list=[]
    data_list.append(digit(servo))
    data_list.append(digit(float(dict[0]+1)))

    return data_list


    return data_list

####################첫번째 유전자들 탄생@1
def generate_population(size):
    population = []
    for i in range(size):
        population.append(receive_val1())
    return population

# k=generate_population(5)
# print(k)



####################유전자 점수 측정, 정렬, 선택
#점수 측정
def fitness(target_score, guess_score):
    return digit(abs(target_score-guess_score))

#점수순서대로 유전자 정렬@2
def compute_performace(population, target_score):
    population_list = []
    for individual in population:
        score = fitness(target_score, individual[1])
        population_list.append([individual, score])
        population_sorted = sorted(population_list, key=lambda x:x[1])
    return population_sorted

# p=compute_performace(k,target_score)
# print(p)


#다음세대로 갈 유전자 선택@3
def select_survivors(population_sorted, num_best, new_data, target_score):
    next_generation =[]

    for i in range(num_best):
        next_generation.append(population_sorted[i][0])

    for j in range(new_data):
        next_generation.append(receive_val1())

    random.shuffle(next_generation)
    return next_generation


# s=select_survivors(p,3,2,target_score)
# print(s)

####################다음세대 유전자 생성
#유전자 생성방법
def create_child(individual):
    child = []
    while True:
        ran=random.random()*3
        if ran<1:
            individual[0]-=10
        elif ran<2:
            individual[0]+=10

        if individual[0] >= min_servo and individual[0] <= max_servo:
            break

    child = receive_val2(individual[0])
    return child
# print(create_child([10,-10,15,10]))

#유전자들 생성@4
def create_children(parents):
    next_population =[]
    for i in range(int(len(parents))):
        next_population.append(create_child(parents[i]))
    return next_population

# children=create_children(s)
# print(children)

####################돌연변이 생성

#돌연변이 생성
def mutate_child(individual):
    child = []
    individual[0] = random.randint(min_servo, max_servo)

    child = receive_val2(individual[0])
    return child

#단체 돌연변이 생성 후 다음세대 확정@5
def mutate_population(population, chance_of_mutation):
    for i in range(len(population)):
        if random.random()*100 < chance_of_mutation:
            population[i]=mutate_child(population[i])
    return population

# new_generation = mutate_population(children, 10)
# print(new_generation)



############실행
n_generation = 100 #최대 세대
population = 5
num_best = 3
chance_of_mutation = 30
target_score = int(input("1~5입력:"))
max_servo=160
min_servo=95

pop = generate_population(population)
for g in range(n_generation):
    # print("\n",pop)
    print("==================== ", g + 1, "세대 ====================")
    sorted_list=compute_performace(pop, target_score)
    if float(sorted_list[0][1])==0:
        print("발견했습니다.",sorted_list[0][0])
        break

    survivors = select_survivors(sorted_list, num_best, population-num_best, target_score)
    children = create_children(survivors)
    new_generation = mutate_population(children, chance_of_mutation)
    pop = new_generation


    for i in range (3):
        print(sorted_list[i])




