import qdbc

host = "localhost"
port = 20101
wrong_port = 12345

try:
    qdbc.query()
except:
    print "expected failure"

try:
    qdbc.query(host, wrong_port, "asdf")
except:
    print "expected failure"

atoms = ["0b", "42", "`asd", "2017.12.08"]

for atom in atoms:
    print qdbc.query(host, port, atom)

