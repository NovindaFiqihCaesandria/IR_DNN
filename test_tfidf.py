from tfidf import TfIdf
import unittest


class TestTfIdf(unittest.TestCase):
    def test_similarity(self):
        table = TfIdf()
        f = open("train_kuhp.txt", "r")
        ft = open("kuhp_test.txt", "r")
        kata_list = []
        res = []
        lab = []
        temp = 0
        for line in f:
            temp=temp+1
            print(str(temp))
            msg = line.strip().split(";")


            kalimat = msg[3].strip().split("/s")
            table.add_document(msg[2], kalimat)
            # if (temp == msg[0]):
            #     kata_list.append(msg[1])
            # elif (temp != msg[0]):
            #     print(msg[0])
            #     msg[0] = temp
            #     dlist.append(msg[0])
            #     # dlist1.append(kata_list)
            #     kata_list = []
            #
            #     # kata_list.append(msg[1])
            # elif (temp == 0):
            #     msg[0] = temp
            #     kata_list.append(msg[1])
        for line in ft:
            msg = line.strip().split(".,")
            lab.append(msg[1])
            kalimat = msg[0].strip().split(" ")
            res.append(table.similarities(kalimat))

        # print(dlist)
        # print(dlist1)

        # for i in range(len(dlist)):
        #  table.add_document(dlist[i], dlist1[1])

        # self.assertEqual(
        #     table.similarities(["a", "b", "c"]),
        #     [["foo", 0.6875], ["bar", 0.75], ["baz", 0.0]])


if __name__ == "__main__":
    unittest.main()
