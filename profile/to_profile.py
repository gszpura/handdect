from pdym.main import mainSubHSV


def run_profile(function):
	try:
		function(profile=50)
	except Exception as e:
		print "Exception: ", e


run_profile(mainSubHSV)