# Provides a base class for running several jobs in parallel, controlling them
# while they run, etc.

import multiprocessing
from multiprocessing import Process
import time

class PoolWorker(Process):
    """
    A class for a process that is part of a scatter pool, where jobs are given
    to pool workers through a shared queue, and commands for suspending/
    resuming/ending execution in a worker can be sent to workers through a
    separate channel.
    
    To use the class, subclass it and override job() and maybe pre_run()
    """
    def __init__(self, tasks, command_pipe, results_queue=None):
        self._t = tasks
        self._out_q = results_queue
        self._cmd = command_pipe
        Process.__init__(self)

    def run(self):
        self.pre_run()
        while True:
            if self._t.empty():
                time.sleep(0.001)
                continue
             
            # Before each simulation, check for orders from the central process
            if self._cmd.poll():
                cmd = self._cmd.recv()
                if cmd == "end":
                    return
            
            res = self.job(self._t.get())
            if self._out_q is not None:
                self._out_q.put(res)
                del res # Allow returning huge results.
        print self.pid, "returning"
        print "***************************"
                     
    def pre_run(self):
        """Override to do something before the jobs-loop starts"""
        pass

    def job(self, prm):
        """
        Override to do the job whose parameters are in prm, as taken from the
        jobs queue.
        """
        pass

import platform
on_windows = (platform.system() == 'Windows')

if not on_windows:
	import os
	import signal
	import cmd
	import sys
	import select
	import tty
	import termios
	import Queue as queue # To avoid confusion with multiprocessing.Queue

	class CLIController(cmd.Cmd):
		sleep_time = 0.1
		
		def __init__(self, process_list,
			completekey=None, stdin=sys.stdin, stdout=sys.stdout):
			"""
			process_list - a list of tuples (process, pipe-entrance)
			completekey, stdin, stdout - passed on to cmd.Cmd
			"""
			self._pl = process_list
			self._in = stdin
			self._out = stdout
			cmd.Cmd.__init__(self, completekey, stdin, stdout)
			
		def do_list(self, line):
			for proc in self._pl:
				self._out.write(proc[0].name + "\n")
		
		def do_quit(self, line):
			for proc in self._pl:
				proc[1].send('end')
		
		def do_terminate(self, line):
			for proc in self._pl:
				proc[0].terminate()
		
		def do_enable(self, who):
			for proc in self._pl:
				if (proc[0].name == who) or (who == "all"):
					os.kill(proc[0].pid, signal.SIGCONT)
		
		def do_disable(self, who):
			for proc in self._pl:
				if (proc[0].name == who) or (who == "all"):
					os.kill(proc[0].pid, signal.SIGSTOP)
		
		def listen_loop(self, results_queue=None, callback=None, comm_callback=None):
			"""
			A loop that terminates when no more jobs are being processed,
			and handles output and process messages as it comes.
			
			Arguments:
			results_queue - poll this queue for results.
			callback - call this when a result arrives from a child process.
			comm_callback - call this when a process sends a message on the control
				pipe. Signature: callback(message). The result is sent back to the 
				asking process on the command pipe.
			"""
			# Thanks to Graham King for the example
			# http://www.darkcoding.net/software/non-blocking-console-io-is-not-possible/
			
			# Save terminal's blocking mode and go non-blocking
			old_settings = termios.tcgetattr(self._in)
			tty.setcbreak(self._in.fileno(), termios.TCSANOW)
			
			self._out.write(self.prompt)
			self._out.flush()
			
			# Just like old-times in BASIC:
			inp = ''
			try:
				while True:
					has_inp = select.select([self._in], [], [], 0)
					if self._in in has_inp[0]:
						# Read char-by-char and show back the typing:
						c = sys.stdin.read(1)
						self._out.write(c)
						
						# On newline interpret line, otherwise keep accumulating:
						if c == "\n":
							self.onecmd(inp)
							inp = ""
							self._out.write(self.prompt)
						else:
							inp += c
						self._out.flush()
					
					# Process results as they arrive:
					if results_queue is not None:
						try:
							r = results_queue.get(False)
							callback(r)
						except queue.Empty:
							pass
					
					# Communicate with individual processes:
					# Note that this blocks the loop until results are sent, so
					# it's not really for long stuff. A non-blocking version is
					# left for the future.
					if comm_callback is not None:
						for proc in self._pl:
							if proc[1].poll():
								res = comm_callback(proc[1].recv())
								proc[1].send(res)
					
					# Break if all processes terminated:
					if len(multiprocessing.active_children()) == 0:
						break
					
					time.sleep(self.sleep_time)
			finally:
				# Command-loop terminated, restore terminal settings:
				termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
			
			if results_queue is None:
				return
			
			# Keep collecting output after commandline finished.
			# This will happen if children finished normally, or if 'terminate' was
			# used while some results weren't handled yet, which is pretty much the
			# same condition.
			try:
				while True:
					r = results_queue.get(False)
					callback(r)
			except queue.Empty:
				pass
