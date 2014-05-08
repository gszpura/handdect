from pdym.main import main_system


def run_profile(function):
	try:
		function(profile=50)
	except Exception as e:
		print "Exception: ", e


run_profile(main_system)