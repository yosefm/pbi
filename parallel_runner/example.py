# Example for usage of the PoolWorker class.

from parallel_runner import PoolWorker, CLIController
import time

class Exhibitionist(PoolWorker):
	def job(self, prm):
		print self.name + "in da house! got " + str(prm)
		print "Thinking hard..."
		time.sleep(5) # Just to show what happens on long runs
		return (prm, prm**2)

if __name__ == "__main__":
	from multiprocessing import Pipe, Queue
	import sys
	
	tasks = Queue()
	out = Queue()
	for num in xrange(20):
		tasks.put(num)
	time.sleep(1)
	
	# Run workers:
	w = []
	for p in xrange(4):
		pside, cside = Pipe()
		t = Exhibitionist(tasks, cside, out)
		w.append((t, pside))
		t.start()
		
		time.sleep(0.1)
	
	# Command line that quits when the last process quits.
	ctrl = CLIController(w)
	ctrl.prompt = "sim> "
	ctrl.listen_loop(out,
		lambda r: sys.stdout.write("Slave says %d**2 = %d\n" % (r[0], r[1])))
	
	# Wait for everyone to finish:
	print "\nWaiting for jobs to finish..."
	for proc in w:
		proc[0].join()
