from pokemontcgsdk import Card
import pokemontcgsdk

pokemontcgsdk.RestClient.configure('0d639dec-7696-4134-9f1b-c2366d85b56d')

results = Card.where(q='name:Gyarados')
print(len(results))
for c in results:
    print(c.name, c.number, c.set.id)