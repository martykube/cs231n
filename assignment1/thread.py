import threading
import Queue
import time

best_net = None # store the best model into this

def worker(queue):
    done = False
    print "starting..."
    while not done:
        try:
            epocs, hidden, learning, reg = queue.get_nowait()
            print "epocs %d, hidden %d, learning %1.1e, reg %1.1e" % (epocs, hidden, learning, reg)
            queue.task_done()
        except Queue.Empty:
            done = True
    print "exiting"


q = Queue.Queue()
for i in range(int(1e4)):
    q.put((i, 0, 0, 0))
    
for w in range(4):
    t = threading.Thread(target=worker, args=(q, ))
    t.daemon = True
    t.start()

q.join()
print 'Yo!'
