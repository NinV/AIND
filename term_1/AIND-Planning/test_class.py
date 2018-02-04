from aimacode.planning import Action
from aimacode.search import Problem
from aimacode.utils import expr
from lp_utils import decode_state
from my_planning_graph import PgNode_a, PgNode_s


def get_actions():
    precond_pos = [expr("Have(Cake)")]
    precond_neg = []
    effect_add = [expr("Eaten(Cake)")]
    effect_rem = [expr("Have(Cake)")]
    eat_action = Action(expr("Eat(Cake)"),
                        [precond_pos, precond_neg],
                        [effect_add, effect_rem])
    precond_pos = []
    precond_neg = [expr("Have(Cake)")]
    effect_add = [expr("Have(Cake)")]
    effect_rem = []
    bake_action = Action(expr("Bake(Cake)"),
                         [precond_pos, precond_neg],
                         [effect_add, effect_rem])
    return [eat_action, bake_action]

actions = get_actions()



a_node_1 = PgNode_a(actions[0])
a_node_2 = PgNode_a(actions[0])
a_node_3 = PgNode_a(actions[0])

s_node_1 = PgNode_s("eaten", True)
s_node_2 = PgNode_s("eaten", True)
s_node_3 = PgNode_s("eaten", True)

a_nodes = set()
s_nodes = set()


print(len(a_nodes), len(s_nodes))

a_node_3.children.add(s_node_3)
s_node_3.parents.add(a_node_3)
a_nodes.add(a_node_3)
s_nodes.add(s_node_3)

s_node_3.show()
a_node_3.show()



for a_node in a_nodes:
    a_node.show()
    for children in a_node.children:
        children.show()

