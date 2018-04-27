@app.route('/interactive/')
def interactive():
	try:
		return render_template('interactive.html')
	except Exception as e:
		return (str(e))

@app.route('/background_process')
def background_process():
	try:
		lang = request.args.get('proglang', 0, type=str)
		if lang.lower() == 'python':
			return jsonify(result='You are wise')
		else:
			return jsonify(result='Try again.')
	except Exception as e:
		return str(e)

@app.route('/pygalexample/')
def pygalexample():
	try:
		graph = pygal.Line()
		graph.title = '% Change Coolness of programming languages over time.'
		graph.x_labels = ['2011','2012','2013','2014','2015','2016']
		graph.add('Python',  [15, 31, 89, 200, 356, 900])
		graph.add('Java',    [15, 45, 76, 80,  91,  95])
		graph.add('C++',     [5,  51, 54, 102, 150, 201])
		graph.add('All others combined!',  [5, 15, 21, 55, 92, 105])
		graph_data = graph.render_data_uri()
		return render_template('graphing.html', graph_data=graph_data)
	except Exception as e:
		return (str(e))

@app.route('/send-mail/')
def send_mail():
    try:
    	msg = Message('Send Mail Tester',
    		sender='yoursendingemail@gmail.com',
    		recipients=['recievingemail@email.com'])
    	msg.body = 'This is the Tester Mail'
    	mail.send(msg)
    	return 'Mail Sent!!!'

    except Exception as e:
    	return str(e)

@app.route('/secret/<path:filename>')
@special_requirement
def protected(filename):
	try:
		return send_from_directory(os.path.join(app.instance_path,''), filename)

	except Exception as e:
		return redirect(url_for('homepage'))