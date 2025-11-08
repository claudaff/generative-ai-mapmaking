import gradio as gr
import os
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import zipfile
import config
import einops
import torch
import random
from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
import cv2

# Python script implementing the web-based application for map tile generation

model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict('Combined.ckpt', location='cuda'), strict=False)  # adjust path to model
model = model.cuda()
ddim_sampler = DDIMSampler(model)

image_resolution = 512
ddim_steps = 20
guess_mode = False
strength = 1
scale = 9
seed = 1286028432
eta = 0
a_prompt = 'best quality, extremely detailed'
n_prompt = ''
num_samples = 1

# HTML and JavaScript for the canvas and interactive elements
custom_canvas_html = """

<h3>Step 2: Select a map style</h3>
<button id="chooseSwisstopo">Swisstopo</button>
<br>
<button id="chooseSiegfried">Siegfried</button>
<br>
<button id="chooseOldNational">Old National</button>
<br>
<h3 id="multiTile" style="display: none;">Step 3: Upload multiple vector maps at once</h3>
<div id="singleTile" style="display: none;">
  <h3>Step 3: Upload vector map or draw the features yourself</h3>
  <div id="container" style="display: grid;">
    <canvas id="drawingCanvas" width="512px" height="512px" style="border:1px solid #000000;grid-area: 1 / 1; margin-bottom: 1%" ;></canvas>
    <br>
    <canvas id="overlayCanvas" width="512" height="512" style="display: none; border:1px solid #000000;grid-area: 1 / 1;"></canvas>
    <div id="containerStyle" style="grid-area: 1 / 3; align-items: center;">
      <div id="colorChoice" style="display: none; flex-wrap: wrap;">
        <button id="background" style="display: block;">Background</button>
        <button id="forest" style="display: block;">Forest</button>
        <button id="tree" style="display: block;">Tree</button>
        <button id="lake" style="display: block;">Lake</button>
        <button id="river" style="display: block;">River</button>
        <button id="stream" style="display: block;">Stream</button>
        <button id="building" style="display: block;">Building</button>
        <button id="road" style="display: block;">Road</button>
        <button id="highway" style="display: block;">Highway</button>
        <button id="through" style="display: block;">Through Road</button>
        <button id="connect" style="display: block;">Connecting Road</button>
        <button id="path" style="display: block;">Path</button>
        <button id="railwaysingle" style="display: block;">Railway (Single Track)</button>
        <button id="railwaymultiple" style="display: block;">Railway (Double Track)</button>
        <input type="range" min="7" max="80" value="7" class="slider" id="myRange" style="display: none;">
      </div>
      <input type="range" min="7" max="80" value="7" class="slider" id="myRange" style="display: none;">
    </div>
  </div>
  <input type="file" id="uploadImage" accept="image/*">
  <button id="clearButton">Clear Canvas</button>
</div>
"""

# Since Gradio only allows connecting the application with a single JavaScript function everything had to be put into one large function called 'webappFunctionalities'. No helper functions could be used either which led to repetitions.

js = """
function webappFunctionalities() {

	var canvas = document.getElementById('drawingCanvas');
	var ctx = canvas.getContext('2d');
	ctx.fillStyle = "white";
	ctx.fillRect(0, 0, 512, 512);

	var canvasOver = document.getElementById('overlayCanvas');
	var ctxOver = canvasOver.getContext('2d');
	var canvasRect = canvas.getBoundingClientRect();

	var clearCanvasButton = document.getElementById('clearButton');
	var uploadImageButton = document.getElementById('uploadImage');
	var choice1Button = document.getElementById('chooseSwisstopo');
	var choice2Button = document.getElementById('chooseSiegfried');
	var choice3Button = document.getElementById('chooseOldNational');

	var forestButton = document.getElementById('forest');
	var lakeButton = document.getElementById('lake');
	var roadButton = document.getElementById('road');
	var backgroundButton = document.getElementById('background');
	var riverButton = document.getElementById('river');
	var streamButton = document.getElementById('stream');
	var highwayButton = document.getElementById('highway');
	var throughButton = document.getElementById('through');
	var connectButton = document.getElementById('connect');
	var treeButton = document.getElementById('tree');
	var railButton = document.getElementById('railwaysingle');
	var railsButton = document.getElementById('railwaymultiple');
	var pathButton = document.getElementById('path');
	var buildingsButton = document.getElementById('building');

	const buttonArray = [forestButton, lakeButton, roadButton, backgroundButton, riverButton, streamButton, highwayButton, throughButton, connectButton, treeButton, railButton, railsButton, pathButton, buildingsButton];

	var color = "white";
	var cursorCircle = document.createElement('div');

	let isDrawing = false;
	let isFirstLineDrawn = false;
	let isRectangleDrawn = false;
	let startX, startY, endX, endY;
	let savedEndX, savedEndY;
	var drawingRectangle = false;
	let shapes = [];

	cursorCircle.style.position = 'absolute';
	cursorCircle.style.border = '1px solid black';
	cursorCircle.style.borderRadius = '50%';
	cursorCircle.style.pointerEvents = 'none';
	cursorCircle.style.width = '14px'; // Default size based on initial range value
	cursorCircle.style.height = '14px';
	cursorCircle.style.transform = 'translate(-50%, -50%)';
	document.body.appendChild(cursorCircle);


	document.querySelector('#mode input[name="radio-1"]').addEventListener('click', function() {

		var r1 = document.querySelector('#mode input[name="radio-1"]'); // Single
		var r1_check = r1.getAttribute("aria-checked");



		document.getElementById('singleTile').style.display = 'block';
		document.getElementById('multiTile').style.display = 'none';

	});

	document.querySelector('#mode input[name="radio-2"]').addEventListener('click', function() {

		var r2 = document.querySelector('#mode input[name="radio-2"]'); // Multiple
		var r2_check = r2.getAttribute("aria-checked");

		document.getElementById('singleTile').style.display = 'none';
		document.getElementById('multiTile').style.display = 'block';

		cursorCircle.style.width = `0px`;
		cursorCircle.style.height = `0px`;

	});


	canvasOver.addEventListener('mousedown', (e) => {

		const rectOver = canvasOver.getBoundingClientRect();
		const mouseXOver = e.clientX - rectOver.left;
		const mouseYOver = e.clientY - rectOver.top;

		if (drawingRectangle) {

			if (!isFirstLineDrawn) {
				// Start or complete the initial line
				if (!isDrawing) {
					isDrawing = true;
					startX = mouseXOver;
					startY = mouseYOver;
				} else {
					isDrawing = false;
					isFirstLineDrawn = true;
					savedEndX = mouseXOver;
					savedEndY = mouseYOver;
				}
			} else if (!isRectangleDrawn) {
				// Define the second dimension of the rectangle
				isRectangleDrawn = true;
				//draw(mouseXOver, mouseYOver, true);

				secondX = mouseXOver;
				secondY = mouseYOver;
				fill = true;

				ctxOver.clearRect(0, 0, canvasOver.width, canvasOver.height);

				shapes.forEach(shape => {
					ctxOver.beginPath();
					ctxOver.moveTo(shape.startX, shape.startY);
					ctxOver.lineTo(shape.endX, shape.endY);
					ctxOver.lineTo(shape.x3, shape.y3);
					ctxOver.lineTo(shape.x4, shape.y4);
					ctxOver.closePath();
					if (shape.fill) {
						ctxOver.strokeStyle = "rgb(82, 82, 82)";
						ctxOver.fillStyle = 'rgb(82, 82, 82)';
						ctxOver.fill();
					}
					ctxOver.stroke();
				});

				// Draw the current shape
				ctxOver.beginPath();
				ctxOver.moveTo(startX, startY);
				ctxOver.lineTo(endX, endY);
				ctxOver.stroke();

				if (isFirstLineDrawn) {
					// Calculate the angle of the initial line
					const angle = Math.atan2(endY - startY, endX - startX);

					// Calculate the length of the initial line
					const length = Math.sqrt((endX - startX) ** 2 + (endY - startY) ** 2);

					// If secondX and secondY are provided, calculate the width based on mouse movement
					if (secondX !== null && secondY !== null) {
						// Calculate the perpendicular distance (width) from the mouse point to the initial line
						const dx = secondX - savedEndX;
						const dy = secondY - savedEndY;
						const projectionLength = (dx * (endX - startX) + dy * (endY - startY)) / length;
						const perpendicularLength = Math.sqrt(dx * dx + dy * dy - projectionLength * projectionLength);

						// Determine the sign of the width based on the relative position of the mouse
						const sign = (dx * (endY - startY) - dy * (endX - startX)) > 0 ? 1 : -1;
						const width = sign * perpendicularLength;

						// Calculate the corners of the rectangle
						const x3 = savedEndX + width * Math.sin(angle);
						const y3 = savedEndY - width * Math.cos(angle);
						const x4 = startX + width * Math.sin(angle);
						const y4 = startY - width * Math.cos(angle);

						// Draw and fill the rectangle
						ctxOver.beginPath();
						ctxOver.moveTo(startX, startY);
						ctxOver.lineTo(endX, endY);
						ctxOver.lineTo(x3, y3);
						ctxOver.lineTo(x4, y4);
						ctxOver.closePath();

						if (fill) {
							ctxOver.strokeStyle = "rgb(82, 82, 82)";
							ctxOver.fillStyle = 'rgb(82, 82, 82)'; // Fill color
							ctxOver.fill();
							isDrawing = false;
							isFirstLineDrawn = false;
							isRectangleDrawn = false;

							// Store the rectangle in the shapes array
							shapes.push({
								startX,
								startY,
								endX,
								endY,
								x3,
								y3,
								x4,
								y4,
								fill
							});
						}

						ctxOver.stroke();
					}
				}



			}

		} else {

		}



	});

	canvas.addEventListener('mousedown', (e) => {
		const rect = canvas.getBoundingClientRect();
		const mouseX = e.clientX - rect.left;
		const mouseY = e.clientY - rect.top;

		if (drawingRectangle) {


		} else {
			isDrawing = true;
			ctxOver.strokeStyle = color;
			ctx.lineWidth = document.getElementById('myRange').value;
			ctx.beginPath();
			ctx.moveTo(mouseX, mouseY);
		}

	});

	canvas.addEventListener('mouseup', function() {

		if (!drawingRectangle) {

			isDrawing = false;
		}
	});

	canvasOver.addEventListener('mousemove', (e) => {

		cursorCircle.style.left = `${e.pageX}px`;
		cursorCircle.style.top = `${e.pageY}px`;
		cursorCircle.style.width = `0px`;
		cursorCircle.style.height = `0px`;

		if (!isDrawing && !isFirstLineDrawn) return;

		const rectOver = canvasOver.getBoundingClientRect();
		const mouseXOver = e.clientX - rectOver.left;
		const mouseYOver = e.clientY - rectOver.top;

		if (isDrawing) {


			if (drawingRectangle) {

				endX = mouseXOver;
				endY = mouseYOver;
				secondX = null;
				secondY = null;
				fill = false;

				ctxOver.clearRect(0, 0, canvasOver.width, canvasOver.height);

				shapes.forEach(shape => {
					ctxOver.beginPath();
					ctxOver.moveTo(shape.startX, shape.startY);
					ctxOver.lineTo(shape.endX, shape.endY);
					ctxOver.lineTo(shape.x3, shape.y3);
					ctxOver.lineTo(shape.x4, shape.y4);
					ctxOver.closePath();
					if (shape.fill) {
						ctxOver.strokeStyle = "rgb(82, 82, 82)";
						ctxOver.fillStyle = 'rgb(82, 82, 82)';
						ctxOver.fill();
					}
					ctxOver.stroke();
				});

				// Draw the current shape
				ctxOver.beginPath();
				ctxOver.moveTo(startX, startY);
				ctxOver.lineTo(endX, endY);
				ctxOver.stroke();

				if (isFirstLineDrawn) {
					// Calculate the angle of the initial line
					const angle = Math.atan2(endY - startY, endX - startX);

					// Calculate the length of the initial line
					const length = Math.sqrt((endX - startX) ** 2 + (endY - startY) ** 2);

					// If secondX and secondY are provided, calculate the width based on mouse movement
					if (secondX !== null && secondY !== null) {
						// Calculate the perpendicular distance (width) from the mouse point to the initial line
						const dx = secondX - savedEndX;
						const dy = secondY - savedEndY;
						const projectionLength = (dx * (endX - startX) + dy * (endY - startY)) / length;
						const perpendicularLength = Math.sqrt(dx * dx + dy * dy - projectionLength * projectionLength);

						// Determine the sign of the width based on the relative position of the mouse
						const sign = (dx * (endY - startY) - dy * (endX - startX)) > 0 ? 1 : -1;
						const width = sign * perpendicularLength;

						// Calculate the corners of the rectangle
						const x3 = savedEndX + width * Math.sin(angle);
						const y3 = savedEndY - width * Math.cos(angle);
						const x4 = startX + width * Math.sin(angle);
						const y4 = startY - width * Math.cos(angle);

						// Draw and fill the rectangle
						ctxOver.beginPath();
						ctxOver.moveTo(startX, startY);
						ctxOver.lineTo(endX, endY);
						ctxOver.lineTo(x3, y3);
						ctxOver.lineTo(x4, y4);
						ctxOver.closePath();

						if (fill) {
							ctxOver.strokeStyle = "rgb(82, 82, 82)";
							ctxOver.fillStyle = 'rgb(82, 82, 82)'; // Fill color
							ctxOver.fill();
							isDrawing = false;
							isFirstLineDrawn = false;
							isRectangleDrawn = false;

							// Store the rectangle in the shapes array
							shapes.push({
								startX,
								startY,
								endX,
								endY,
								x3,
								y3,
								x4,
								y4,
								fill
							});
						}

						ctxOver.stroke();
					}
				}



			} else {

			}




		} else if (isFirstLineDrawn && !isRectangleDrawn) {
			//draw(mouseXOver, mouseYOver);

			secondX = mouseXOver;
			secondY = mouseYOver;
			fill = false;

			ctxOver.clearRect(0, 0, canvasOver.width, canvasOver.height);

			shapes.forEach(shape => {
				ctxOver.strokeStyle = "rgb(82, 82, 82)";
				ctxOver.beginPath();
				ctxOver.moveTo(shape.startX, shape.startY);
				ctxOver.lineTo(shape.endX, shape.endY);
				ctxOver.lineTo(shape.x3, shape.y3);
				ctxOver.lineTo(shape.x4, shape.y4);
				ctxOver.closePath();
				if (shape.fill) {
					ctxOver.strokeStyle = "rgb(82, 82, 82)";
					ctxOver.fillStyle = 'rgb(82, 82, 82)';
					ctxOver.fill();
				}
				ctxOver.stroke();
			});

			// Draw the current shape
			ctxOver.beginPath();
			ctxOver.moveTo(startX, startY);
			ctxOver.lineTo(endX, endY);
			ctxOver.stroke();

			if (isFirstLineDrawn) {
				// Calculate the angle of the initial line
				const angle = Math.atan2(endY - startY, endX - startX);

				// Calculate the length of the initial line
				const length = Math.sqrt((endX - startX) ** 2 + (endY - startY) ** 2);

				// If secondX and secondY are provided, calculate the width based on mouse movement
				if (secondX !== null && secondY !== null) {
					// Calculate the perpendicular distance (width) from the mouse point to the initial line
					const dx = secondX - savedEndX;
					const dy = secondY - savedEndY;
					const projectionLength = (dx * (endX - startX) + dy * (endY - startY)) / length;
					const perpendicularLength = Math.sqrt(dx * dx + dy * dy - projectionLength * projectionLength);

					// Determine the sign of the width based on the relative position of the mouse
					const sign = (dx * (endY - startY) - dy * (endX - startX)) > 0 ? 1 : -1;
					const width = sign * perpendicularLength;

					// Calculate the corners of the rectangle
					const x3 = savedEndX + width * Math.sin(angle);
					const y3 = savedEndY - width * Math.cos(angle);
					const x4 = startX + width * Math.sin(angle);
					const y4 = startY - width * Math.cos(angle);

					// Draw and fill the rectangle
					ctxOver.beginPath();
					ctxOver.moveTo(startX, startY);
					ctxOver.lineTo(endX, endY);
					ctxOver.lineTo(x3, y3);
					ctxOver.lineTo(x4, y4);
					ctxOver.closePath();

					if (fill) {
						ctxOver.strokeStyle = "rgb(82, 82, 82)";
						ctxOver.fillStyle = 'rgb(82, 82, 82)'; // Fill color
						ctxOver.fill();
						isDrawing = false;
						isFirstLineDrawn = false;
						isRectangleDrawn = false;

						// Store the rectangle in the shapes array
						shapes.push({
							startX,
							startY,
							endX,
							endY,
							x3,
							y3,
							x4,
							y4,
							fill
						});
					}

					ctxOver.stroke();
				}
			}


		}
	});

	canvas.addEventListener('mousemove', (e) => {

		cursorCircle.style.left = `${e.pageX}px`;
		cursorCircle.style.top = `${e.pageY}px`;
		cursorCircle.style.width = `${document.getElementById('myRange').value}px`;
		cursorCircle.style.height = `${document.getElementById('myRange').value}px`;

		if (!isDrawing && !isFirstLineDrawn) return;

		const rect = canvas.getBoundingClientRect();
		const mouseX = e.clientX - rect.left;
		const mouseY = e.clientY - rect.top;

		if (isDrawing) {


			if (drawingRectangle) {


			} else {
				ctx.lineTo(mouseX, mouseY);
				ctx.stroke();
				ctx.lineJoin = 'round';
				ctx.lineCap = 'round';
				ctx.lineWidth = document.getElementById('myRange').value;
				ctx.strokeStyle = color;

			}




		} else if (isFirstLineDrawn && !isRectangleDrawn) {

		}
	});

	canvasOver.addEventListener('mouseleave', function() {
		ctx.drawImage(canvasOver, 0, 0);
		isDrawing = false;
		var dataURL = canvas.toDataURL();
		document.getElementById('canvasOutput').value = dataURL;
	});

	canvas.addEventListener('mouseleave', function() {
		isDrawing = false;
		var dataURL = canvas.toDataURL();
		document.getElementById('canvasOutput').value = dataURL;
	});

	buildingsButton.addEventListener('mousedown', function() {

		buttonArray.forEach(function(button_element) {
			if (button_element != buildingsButton) {
				button_element.style.outline = 'none';
			} else {
				button_element.style.outline = '4px solid black';
			}
		});
		if (!drawingRectangle) {
			ctxOver.strokeStyle = "rgb(82, 82, 82)";
			color = "rgb(82, 82, 82)";
			canvasOver.style.display = 'block';
			drawingRectangle = true;
		}
	});


	forestButton.addEventListener('mousedown', function() {

		buttonArray.forEach(function(button_element) {
			if (button_element != forestButton) {
				button_element.style.outline = 'none';
			} else {
				button_element.style.outline = '4px solid black';
			}
		});

		color = "rgb(77, 175, 74 )";
		drawingRectangle = false;
		shapes = [];
		ctx.drawImage(canvasOver, 0, 0);
		canvasOver.style.display = 'none';
		ctxOver.clearRect(0, 0, canvasOver.width, canvasOver.height);
	});


	lakeButton.addEventListener('mousedown', function() {

		buttonArray.forEach(function(button_element) {
			if (button_element != lakeButton) {
				button_element.style.outline = 'none';
			} else {
				button_element.style.outline = '4px solid black';
			}
		});

		color = "rgb(55, 126, 184 )";
		drawingRectangle = false;
		shapes = [];
		ctx.drawImage(canvasOver, 0, 0);
		canvasOver.style.display = 'none';
		ctxOver.clearRect(0, 0, canvasOver.width, canvasOver.height);
	});

	roadButton.addEventListener('mousedown', function() {

		buttonArray.forEach(function(button_element) {
			if (button_element != roadButton) {
				button_element.style.outline = 'none';
			} else {
				button_element.style.outline = '4px solid black';
			}
		});
		color = "rgb(149, 74, 162 )";
		drawingRectangle = false;
		shapes = [];
		ctx.drawImage(canvasOver, 0, 0);
		canvasOver.style.display = 'none';
		ctxOver.clearRect(0, 0, canvasOver.width, canvasOver.height);
	});

	backgroundButton.addEventListener('mousedown', function() {

		buttonArray.forEach(function(button_element) {
			if (button_element != backgroundButton) {
				button_element.style.outline = 'none';
			} else {
				button_element.style.outline = '4px solid black';
			}
		});

		color = "white";
		drawingRectangle = false;
		shapes = [];
		ctx.drawImage(canvasOver, 0, 0);
		canvasOver.style.display = 'none';
		ctxOver.clearRect(0, 0, canvasOver.width, canvasOver.height);
	});

	treeButton.addEventListener('mousedown', function() {

		buttonArray.forEach(function(button_element) {
			if (button_element != treeButton) {
				button_element.style.outline = 'none';
			} else {
				button_element.style.outline = '4px solid black';
			}
		});
		color = "rgb(63, 131, 55 )";
		drawingRectangle = false;
		shapes = [];
		ctx.drawImage(canvasOver, 0, 0);
		canvasOver.style.display = 'none';
		ctxOver.clearRect(0, 0, canvasOver.width, canvasOver.height);
	});

	pathButton.addEventListener('mousedown', function() {
		buttonArray.forEach(function(button_element) {
			if (button_element != pathButton) {
				button_element.style.outline = 'none';
			} else {
				button_element.style.outline = '4px solid black';
			}
		});
		color = "rgb(0, 0, 0 )";
		drawingRectangle = false;
		shapes = [];
		ctx.drawImage(canvasOver, 0, 0);
		canvasOver.style.display = 'none';
		ctxOver.clearRect(0, 0, canvasOver.width, canvasOver.height);
	});

	riverButton.addEventListener('mousedown', function() {
		buttonArray.forEach(function(button_element) {
			if (button_element != riverButton) {
				button_element.style.outline = 'none';
			} else {
				button_element.style.outline = '4px solid black';
			}
		});
		color = "rgb(41, 163, 215 )";
		drawingRectangle = false;
		shapes = [];
		ctx.drawImage(canvasOver, 0, 0);
		canvasOver.style.display = 'none';
		ctxOver.clearRect(0, 0, canvasOver.width, canvasOver.height);
	});

	streamButton.addEventListener('mousedown', function() {
		buttonArray.forEach(function(button_element) {
			if (button_element != streamButton) {
				button_element.style.outline = 'none';
			} else {
				button_element.style.outline = '4px solid black';
			}
		});
		color = "rgb(89, 180, 208 )";
		drawingRectangle = false;
		shapes = [];
		ctx.drawImage(canvasOver, 0, 0);
		canvasOver.style.display = 'none';
		ctxOver.clearRect(0, 0, canvasOver.width, canvasOver.height);
	});

	railButton.addEventListener('mousedown', function() {
		buttonArray.forEach(function(button_element) {
			if (button_element != railButton) {
				button_element.style.outline = 'none';
			} else {
				button_element.style.outline = '4px solid black';
			}
		});
		color = "rgb(219, 30, 42 )";
		drawingRectangle = false;
		shapes = [];
		ctx.drawImage(canvasOver, 0, 0);
		canvasOver.style.display = 'none';
		ctxOver.clearRect(0, 0, canvasOver.width, canvasOver.height);
	});

	railsButton.addEventListener('mousedown', function() {
		buttonArray.forEach(function(button_element) {
			if (button_element != railsButton) {
				button_element.style.outline = 'none';
			} else {
				button_element.style.outline = '4px solid black';
			}
		});
		color = "rgb(144, 20, 28 )";
		drawingRectangle = false;
		shapes = [];
		ctx.drawImage(canvasOver, 0, 0);
		canvasOver.style.display = 'none';
		ctxOver.clearRect(0, 0, canvasOver.width, canvasOver.height);
	});

	throughButton.addEventListener('mousedown', function() {
		buttonArray.forEach(function(button_element) {
			if (button_element != throughButton) {
				button_element.style.outline = 'none';
			} else {
				button_element.style.outline = '4px solid black';
			}
		});
		color = "rgb(255, 103, 227 )";
		drawingRectangle = false;
		shapes = [];
		ctx.drawImage(canvasOver, 0, 0);
		canvasOver.style.display = 'none';
		ctxOver.clearRect(0, 0, canvasOver.width, canvasOver.height);
	});

	connectButton.addEventListener('mousedown', function() {
		buttonArray.forEach(function(button_element) {
			if (button_element != connectButton) {
				button_element.style.outline = 'none';
			} else {
				button_element.style.outline = '4px solid black';
			}
		});
		color = "rgb(128, 135, 37 )";
		drawingRectangle = false;
		shapes = [];
		ctx.drawImage(canvasOver, 0, 0);
		canvasOver.style.display = 'none';
		ctxOver.clearRect(0, 0, canvasOver.width, canvasOver.height);
	});

	highwayButton.addEventListener('mousedown', function() {
		buttonArray.forEach(function(button_element) {
			if (button_element != highwayButton) {
				button_element.style.outline = 'none';
			} else {
				button_element.style.outline = '4px solid black';
			}
		});
		color = "rgb(247, 128, 30 )";
		drawingRectangle = false;
		shapes = [];
		ctx.drawImage(canvasOver, 0, 0);
		canvasOver.style.display = 'none';
		ctxOver.clearRect(0, 0, canvasOver.width, canvasOver.height);
	});






	clearCanvasButton.addEventListener('mousedown', function() {
		isDrawing = false;
		ctx.clearRect(0, 0, canvas.width, canvas.height);
		ctxOver.clearRect(0, 0, canvas.width, canvas.height);
	});


	choice1Button.addEventListener('mousedown', function() {

		document.getElementById('chooseSwisstopo').style.outline = '4px solid black';
		document.getElementById('chooseSiegfried').style.outline = 'none';
		document.getElementById('chooseOldNational').style.outline = 'none';
		document.getElementById('colorChoice').style.display = 'flex';
		document.getElementById('myRange').style.display = 'block';

		document.getElementById('through').style.display = 'block';
		document.getElementById('tree').style.display = 'block';
		document.getElementById('connect').style.display = 'block';
		document.getElementById('highway').style.display = 'block';

		var txt = document.querySelector('#selectedStyle textarea');
		txt.value = "map in swisstopo style";
		var event1 = new Event('input');
		txt.dispatchEvent(event1);


	});

	choice2Button.addEventListener('mousedown', function() {

		document.getElementById('chooseSwisstopo').style.outline = 'none';
		document.getElementById('chooseSiegfried').style.outline = '4px solid black';
		document.getElementById('chooseOldNational').style.outline = 'none';
		document.getElementById('colorChoice').style.display = 'flex';
		document.getElementById('myRange').style.display = 'block';

		document.getElementById('through').style.display = 'none';
		document.getElementById('tree').style.display = 'none';
		document.getElementById('connect').style.display = 'none';
		document.getElementById('highway').style.display = 'none';

		var txt = document.querySelector('#selectedStyle textarea');
		txt.value = "map in siegfried style";
		var event2 = new Event('input');
		txt.dispatchEvent(event2);

	});

	choice3Button.addEventListener('mousedown', function() {

		document.getElementById('chooseSwisstopo').style.outline = 'none';
		document.getElementById('chooseOldNational').style.outline = '4px solid black';
		document.getElementById('chooseSiegfried').style.outline = 'none';
		document.getElementById('colorChoice').style.display = 'flex';
		document.getElementById('myRange').style.display = 'block';

		document.getElementById('through').style.display = 'block';
		document.getElementById('tree').style.display = 'none';
		document.getElementById('connect').style.display = 'block';
		document.getElementById('highway').style.display = 'block';

		var txt = document.querySelector('#selectedStyle textarea');
		txt.value = "map in old national style";
		var event3 = new Event('input');
		txt.dispatchEvent(event3);

	});

	uploadImageButton.addEventListener('change', function(event) {
		var file = event.target.files[0];
		if (file) {
			var reader = new FileReader();
			reader.onload = function(e) {
				var img = new Image();
				img.onload = function() {
					ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
				};
				img.src = e.target.result;
			};
			reader.readAsDataURL(file);
		}
	});


}
"""

# Paths for images used in CSS
image_path = 'swisstopoAPP.png'
absolute_path = os.path.abspath(image_path)
image_path2 = 'siegfriedAPP.png'
absolute_path2 = os.path.abspath(image_path2)
image_path3 = 'oldNationalAPP.png'
absolute_path3 = os.path.abspath(image_path3)

css = """
#chooseSwisstopo {
  background: url("file=swisstopoAPP.png");
  font-weight: bold;
  font-size: 24px;
  width: 100%;
  border-radius: 12px;
  border: 1px solid gray;
  margin-top: 0.5%;
  text-shadow: 1px 1px 2px white;
}
#chooseSiegfried {
  background: url("file=siegfriedAPP.png");
  font-weight: bold;
  font-size: 24px;
  width: 100%;
  border-radius: 12px;
  border: 1px solid gray;
  margin-top: 0.5%;
  text-shadow: 1px 1px 2px white;
}
#chooseOldNational {
  background: url("file=oldNationalAPP.png");
  font-weight: bold;
  font-size: 24px;
  width: 100%;
  border-radius: 12px;
  border: 1px solid gray;
  margin-top: 0.5%;
  text-shadow: 1px 1px 2px white;
}

#forest {
  background-color: #ece8e7;
  padding: 10px 32px;
  border: 1px solid black;
}
#lake {
  background-color: #ece8e7;
  padding: 10px 32px;
  border: 1px solid black;
}
#river {
  background-color: #ece8e7;
  padding: 10px 32px;
  border: 1px solid black;
}
#stream {
  background-color: #ece8e7;
  padding: 10px 32px;
  border: 1px solid black;
}
#railwaysingle {
  background-color: #ece8e7;
  padding: 10px 32px;
  border: 1px solid black;
}
#railwaymultiple {
  background-color: #ece8e7;
  padding: 10px 32px;
  border: 1px solid black;
}
#building {
  background-color: #ece8e7;
  padding: 10px 32px;
  border: 1px solid black;
}
#highway {
  background-color: #ece8e7;
  padding: 10px 32px;
  border: 1px solid black;
}
#path {
  background-color: #ece8e7;
  padding: 10px 32px;
  border: 1px solid black;
}
#tree {
  background-color: #ece8e7;
  padding: 10px 32px;
  border: 1px solid black;
}
#through {
  background-color: #ece8e7;
  padding: 10px 32px;
  border: 1px solid black;
}
#connect {
  background-color: #ece8e7;
  padding: 10px 32px;
  border: 1px solid black;
}
#road {
  background-color: #ece8e7;
  padding: 10px 32px;
  border: 1px solid black;
}
#background {
  background-color: #ece8e7;
  padding: 10px 32px;
  border: 1px solid black;
}

#colorChoice button {
  width: 35%;
  margin: 1%;
}

#uploadImage {
}

#clearButton {
}

.slider {
  -webkit-appearance: none;
  width: 35%;
  height: 15px;
  border-radius: 5px;
  background: #d3d3d3;
  outline: none;
  opacity: 0.7;
  -webkit-transition: 0.2s;
  transition: opacity 0.2s;
  margin-top: 2.5%;
  margin-left: 1%;
}

.slider:hover {
  opacity: 1;
}

"""

background_cover = np.array(Image.open("oldNatBackground.png"))[:, :, :3]  # for post-processing Old National


# Function to process single uploaded or drawn vector map on canvas
def process_canvas(data_url, p):
    image_resolution = 512
    ddim_steps = 20
    guess_mode = False
    strength = 1
    scale = 9
    eta = 0
    a_prompt = 'best quality, extremely detailed'
    n_prompt = ''
    prompt = p

    if ('swisstopo' in prompt) or ('national' in prompt):

        seed = 1286028432
        num_samples = 1

    elif 'siegfried' in prompt:

        seed = -1
        num_samples = 1

    else:

        print("ERROR")

    if isinstance(data_url, str):  # if SINGLE image (uploaded or edited)

        generated_tile = []

        # Decode the base64 image
        image_data = base64.b64decode(data_url.split(',')[1])
        image = Image.open(BytesIO(image_data))

        # Process the image (convert to numpy array for example)
        input_image = np.array(image)

        with torch.no_grad():

            img = resize_image(HWC3(input_image), image_resolution)
            H, W, C = img.shape

            control = torch.from_numpy(img.copy()).float().cuda() / 255.0
            control = torch.stack([control for _ in range(num_samples)], dim=0)
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()

            if seed == -1:
                seed = random.randint(0, 65535)
            seed_everything(seed)

            if config.save_memory:
                model.low_vram_shift(is_diffusing=False)

            cond = {"c_concat": [control],
                    "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
            un_cond = {"c_concat": None if guess_mode else [control],
                       "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
            shape = (4, H // 8, W // 8)

            if config.save_memory:
                model.low_vram_shift(is_diffusing=True)

            model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else (
                    [strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
            samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                         shape, cond, verbose=False, eta=eta,
                                                         unconditional_guidance_scale=scale,
                                                         unconditional_conditioning=un_cond)

            if config.save_memory:
                model.low_vram_shift(is_diffusing=False)

            x_samples = model.decode_first_stage(samples)
            x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0,
                                                                                                               255).astype(
                np.uint8)

            results = [x_samples[i] for i in range(num_samples)]

        if 'siegfried' in prompt:

            generated_tile.append(results[0])  # no augmentation taking place

        elif 'swisstopo' in prompt:  # post-processing for Swisstopo

            lower_bound = np.array([255, 255, 255])
            upper_bound = np.array([255, 255, 255])
            background = cv2.inRange(input_image[:, :, :3], lower_bound, upper_bound)
            background = cv2.merge((background, background, background))
            tile = np.where(background == [255, 255, 255], [255, 255, 255], results[0])

            lower_bound = np.array([77, 175, 74])
            upper_bound = np.array([77, 175, 74])
            forest = cv2.inRange(input_image[:, :, :3], lower_bound, upper_bound)
            forest = cv2.merge((forest, forest, forest))
            tile = np.where(forest == [255, 255, 255], [205, 230, 190], tile)

            lower_bound = np.array([82, 82, 82])
            upper_bound = np.array([82, 82, 82])
            buildings = cv2.inRange(input_image[:, :, :3], lower_bound, upper_bound)
            buildings = cv2.merge((buildings, buildings, buildings))
            processedtile = np.where(buildings == [255, 255, 255], [0, 0, 0], tile)

            generated_tile.append(processedtile)

        elif 'national' in prompt:  # post-processing for Old National

            lower_bound = np.array([255, 255, 255])
            upper_bound = np.array([255, 255, 255])
            background = cv2.inRange(input_image[:, :, :3], lower_bound, upper_bound)
            background = cv2.merge((background, background, background))
            processed_tile = np.where(background == 255, background_cover, results[0])

            generated_tile.append(processed_tile)

    return generated_tile


# Function to process multiple uploaded vector maps
def process(input_control, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps, guess_mode, strength,
            scale, seed, eta):
    generated_tiles = []

    if ('swisstopo' in prompt) or ('national' in prompt):

        seed = 1286028432
        num_samples = 1

    elif 'siegfried' in prompt:

        seed = -1
        num_samples = 1

    else:

        print("ERROR")

    for file_obj in input_control:
        # Open and process the image
        with Image.open(file_obj.name) as img_c:

            input_image = np.array(img_c)

            with torch.no_grad():

                img = resize_image(HWC3(input_image), image_resolution)
                H, W, C = img.shape

                control = torch.from_numpy(img.copy()).float().cuda() / 255.0
                control = torch.stack([control for _ in range(num_samples)], dim=0)
                control = einops.rearrange(control, 'b h w c -> b c h w').clone()

                if seed == -1:
                    seed = random.randint(0, 65535)
                seed_everything(seed)

                if config.save_memory:
                    model.low_vram_shift(is_diffusing=False)

                cond = {"c_concat": [control],
                        "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
                un_cond = {"c_concat": None if guess_mode else [control],
                           "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
                shape = (4, H // 8, W // 8)

                if config.save_memory:
                    model.low_vram_shift(is_diffusing=True)

                model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else (
                        [strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
                samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                             shape, cond, verbose=False, eta=eta,
                                                             unconditional_guidance_scale=scale,
                                                             unconditional_conditioning=un_cond)

                if config.save_memory:
                    model.low_vram_shift(is_diffusing=False)

                x_samples = model.decode_first_stage(samples)
                x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0,
                                                                                                                   255).astype(
                    np.uint8)

                results = [x_samples[i] for i in range(num_samples)]

        if 'siegfried' in prompt:

            generated_tiles.append(results[0])  # no augmentation taking place

        elif 'swisstopo' in prompt:

            lower_bound = np.array([255, 255, 255])
            upper_bound = np.array([255, 255, 255])
            background = cv2.inRange(input_image[:, :, :3], lower_bound, upper_bound)
            background = cv2.merge((background, background, background))
            tile = np.where(background == [255, 255, 255], [255, 255, 255], results[0])

            lower_bound = np.array([77, 175, 74])
            upper_bound = np.array([77, 175, 74])
            forest = cv2.inRange(input_image[:, :, :3], lower_bound, upper_bound)
            forest = cv2.merge((forest, forest, forest))
            tile = np.where(forest == [255, 255, 255], [205, 230, 190], tile)

            lower_bound = np.array([82, 82, 82])
            upper_bound = np.array([82, 82, 82])
            buildings = cv2.inRange(input_image[:, :, :3], lower_bound, upper_bound)
            buildings = cv2.merge((buildings, buildings, buildings))
            processedtile = np.where(buildings == [255, 255, 255], [0, 0, 0], tile)

            generated_tiles.append(processedtile)

        elif 'national' in prompt:

            lower_bound = np.array([255, 255, 255])
            upper_bound = np.array([255, 255, 255])
            background = cv2.inRange(input_image[:, :, :3], lower_bound, upper_bound)
            background = cv2.merge((background, background, background))
            processed_tile = np.where(background == 255, background_cover, results[0])
            generated_tiles.append(processed_tile)

    return generated_tiles


def update_visibility(radio):
    value = radio  # Get the selected value from the radio button
    if value == "Single Tile":
        return gr.Button(visible=True), gr.Button(visible=False), gr.File(visible=False), gr.Button(
            visible=False), gr.File(visible=False)
    else:
        return gr.Button(visible=False), gr.Button(visible=True), gr.File(visible=True), gr.Button(
            visible=True), gr.File(visible=True)


output_dir = "output"
os.makedirs(output_dir, exist_ok=True)


def zip_images(output_images):
    images = []

    for element in output_images:
        images.append(element[0])
    print(images)

    zip_path = os.path.join(output_dir, "images.zip")
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for index, img in enumerate(images):
            img_name = f'{index}.png'
            zipf.write(images[index], img_name)
    return zip_path


# Gradio Interface
block = gr.Blocks(css=css, js=js).queue()
with block:
    with gr.Column():
        gr.Markdown("# Map Tile Generator")
        gr.Markdown("### Step 1: Select Generation Mode")
        radio = gr.Radio(["Single Tile", "Multiple Tiles"], label="Generation Mode",
                         info="Single Tile: Draw a vector map on your own or upload and edit it. Multiple Tiles: Upload multiple vector maps.",
                         elem_id="mode")
        canvas_html = gr.HTML(custom_canvas_html)
        upload_files = gr.File(file_count="directory", visible=False)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Step 4: Generate map tile in selected style")
            submit_button = gr.Button("Generate Tile", visible=False)
            prompt = gr.Textbox(label="Selected Map Style", interactive=False, elem_id="selectedStyle")
            submit_button_directory = gr.Button("Generate Tiles", visible=False)
            with gr.Accordion("Advanced options", open=False):
                num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
                image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
                strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                guess_mode = gr.Checkbox(label='Guess Mode', value=False)
                ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=20, step=1)
                scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=9.0, step=0.1)
                seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, randomize=True)
                eta = gr.Number(label="eta (DDIM)", value=0.0)
                a_prompt = gr.Textbox(label="Added Prompt", value='best quality, extremely detailed')
                n_prompt = gr.Textbox(label="Negative Prompt", value='')

        with gr.Column():
            canvas_output = gr.Textbox(visible=False, elem_id="canvasOutput")

            # Add an image output component
            image_output = gr.Gallery(label="Processed Image(s)", format="png", interactive=False)

            zip_button = gr.Button("Zip Images", visible=False)
            download_link = gr.File(label="Download Zipped Images", visible=False)

    zip_button.click(zip_images, inputs=image_output, outputs=download_link)

    radio.change(fn=update_visibility, inputs=radio,
                 outputs=[submit_button, submit_button_directory, upload_files, zip_button, download_link])

    submit_button.click(fn=process_canvas, inputs=[canvas_output, prompt], outputs=[image_output],
                        js="""() => {
                                    var canvas = document.getElementById('drawingCanvas');
                                    var dataURL = canvas.toDataURL();
                                    document.getElementById('canvasOutput').value = dataURL;
                                    const canvasValue =  document.getElementById('canvasOutput').value;
                                    const selectedStyleValue = document.querySelector('#selectedStyle textarea').value;
                                    return [canvasValue, selectedStyleValue]
                        }
                        """
                        )
    ips = [upload_files, prompt, a_prompt, n_prompt, num_samples, image_resolution, ddim_steps,
           guess_mode, strength, scale, seed, eta]
    submit_button_directory.click(fn=process, inputs=ips, outputs=[image_output]
                                  )
block.launch(allowed_paths=[absolute_path, absolute_path2, absolute_path3], share=True)