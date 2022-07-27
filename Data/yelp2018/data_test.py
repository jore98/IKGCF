train_file = 'train.txt'
test_file = 'test.txt'
kg_file = 'kg_final.txt'

n_users, n_items = 0, 0
exist_users = []
n_train, n_test =0, 0

with open(train_file) as f:
    for l in f.readlines():
        if len(l) > 0:
            l = l.strip('\n').split(' ')
            items = [int(i) for i in l[1:]]
            uid = int(l[0])
            exist_users.append(uid)
            n_items = max(n_items, max(items))
            n_users = max(n_users, uid)
            n_train += len(items)

with open(test_file) as f:
    for l in f.readlines():
        if len(l) > 0:
            l = l.strip('\n')
            try:
                items = [int(i) for i in l.split(' ')[1:]]
            except Exception:
                continue
            n_items = max(n_items, max(items))
            n_test += len(items)

n_items += 1
n_users += 1

item2entity = {}
n_kg = 0

with open('kg_final.txt') as f:
    for l in f.readlines():
        if len(l) > 0:
            l = l.strip('\n')
            items = [int(i) for i in l.split(' ')[:]]
            i1, i2 = items[0], items[2]
            n_kg = max(i1, i2, n_kg)

            if i1 < n_items and i2 >= n_items:
                if i1 not in item2entity:
                    item2entity[i1] = []
                item2entity[i1].append(i2-n_items+1)
            if i2 < n_items and i1 >= n_items:
                if i2 not in item2entity:
                    item2entity[i2] = []
                item2entity[i2].append(i1-n_items+1)

item_in_kg = set(item2entity.keys())
for i in range(n_items):
    if i not in item2entity:
        print('item {} not in kg.'.format(i))

n_relation = 0
relation_set = set()
with open('item2relation.txt') as f:
    for l in f.readlines():
        if len(l) > 0:
            l = l.strip('\n')
            try:
                items = [int(i) for i in l.split(' ')[1:]]

            except Exception:
                continue
            relation_set |= set(items)
            n_relation = max(n_relation, max(items))
            print(n_relation)

print(n_relation+1, len(relation_set))