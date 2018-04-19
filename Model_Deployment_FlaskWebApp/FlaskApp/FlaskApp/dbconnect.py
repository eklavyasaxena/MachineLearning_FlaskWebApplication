from flaskext.mysql import MySQL

def connection():
	conn = MySQL.connect()
		# host='localhost',
		# user='root',
		# passwd='predictionsql',
		# db='datascienceflask'
	c = conn.cursor()

	return c, conn