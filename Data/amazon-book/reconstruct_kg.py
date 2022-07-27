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
item2relation = {}
entity2relation = {}
n_kg = 0
n_relations=0

with open('kg_final.txt') as f:
    for l in f.readlines():
        if len(l) > 0:
            l = l.strip('\n')
            items = [int(i) for i in l.split(' ')[:]]
            i1, r, i2 = items[0], items[1], items[2]
            n_kg = max(i1, i2, n_kg)
            if i1 == 136498 or i2 == 136498:
                print(i1, i2)

            if i1 < n_items and i2 >= n_items:
                if i1 not in item2entity:
                    item2entity[i1] = []
                if i1 not in item2relation:
                    item2relation[i1] = []
                if i2 - n_items + 1 not in entity2relation:
                    entity2relation[i2 - n_items + 1] = set()
                item2entity[i1].append(i2 - n_items + 1)
                item2relation[i1].append(r)
                entity2relation[i2 - n_items + 1].add(r)

            if i2 < n_items and i1 >= n_items:
                if i2 not in item2entity:
                    item2entity[i2] = []
                if i2 not in item2relation:
                    item2relation[i2] = []
                if i1 - n_items + 1 not in entity2relation:
                    entity2relation[i1 - n_items + 1] = set()
                item2entity[i2].append(i1 - n_items + 1)
                item2relation[i2].append(r)
                entity2relation[i1 - n_items + 1].add(r)

print(n_kg)
file_e = open('item2entity.txt', 'w')
file_r = open('item2relation.txt', 'w')
file_r2 = open('entity2relation.txt', 'w')

for k, entities in item2entity.items():
    s = '{} '.format(str(k)) + ' '.join([str(i) for i in entities]) + '\n'
    file_e.write(s)

for k, relation in item2relation.items():
    s = '{} '.format(str(k)) + ' '.join([str(i) for i in relation]) + '\n'
    file_r.write(s)

for k, relation in entity2relation.items():
    s = '{} '.format(str(k)) + ' '.join([str(i) for i in relation]) + '\n'
    file_r2.write(s)

file_e.close()
file_r.close()
file_r2.close()
