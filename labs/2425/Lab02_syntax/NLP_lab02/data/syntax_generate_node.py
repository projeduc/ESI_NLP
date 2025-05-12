
def generate_node(node, id=0):
    # If the node does not exist
    if node is None:
        return 0, ''
    # If the node is final
    nid = id + 1
    if (len(node) == 2) and (type(node[1]) == str) :
        return nid, 'N' + str(id) + '[label="' + node[0] + "=" + node[1] + '" shape=box];\n'
    # Otherzise,
    # If there are children, print if else
    res = 'N' + str(id) + '[label="' + node[0] + '"];\n'
    nid_l = nid
    nid, code = generate_node(node[1], id=nid_l)
    res += code
    res += 'N' + str(id) + ' ->  N' + str(nid_l) + ';\n'
    if len(node) > 2:
        nid_r = nid
        nid, code = generate_node(node[2], id=nid_r)
        res += code
        res += 'N' + str(id) + ' ->  N' + str(nid_r) + ';\n'
    return nid, res
