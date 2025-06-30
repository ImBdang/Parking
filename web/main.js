

let toado = []

function read_toado(){
    return fetch('bounding_boxes.json')
    .then(response => response.json())
    .then(data => {
        toado = data
        drawBoxesOnCanvas()
    })
    .catch(error => console.error('Error loading JSON:', error));
}

function load_option(){
    content = ``
    for (let i = 0; i < toado.length; i++){
        template = `<option value="${i}">Slot ${i}</option>\n`
        content += template
 
    }
    document.getElementById("parking_slot").innerHTML = content
}

async function main(){
    await read_toado()
    load_option()
    drawBoxesOnCanvas()
}

function drawBoxesOnCanvas() {
  const canvas = document.getElementById('parkingCanvas');
  const ctx = canvas.getContext('2d');
  
  canvas.width = canvas.offsetWidth;
  canvas.height = canvas.offsetHeight;
  
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  

  toado.forEach(box => {
    if (box.st == true){
        ctx.strokeStyle = '#00FF00'
        ctx.lineWidth = 2;
        ctx.strokeRect(box.x, box.y, box.w, box.h);

        ctx.fillStyle ='rgba(120, 247, 95, 0.5)';
        ctx.fillRect(box.x, box.y, box.w, box.h);
    }
    else {
        ctx.strokeStyle = '#FF0000'
        ctx.lineWidth = 2;
        ctx.strokeRect(box.x, box.y, box.w, box.h);

        ctx.fillStyle ='rgba(204, 32, 32, 0.5)';
        ctx.fillRect(box.x, box.y, box.w, box.h);
    }
  });
}


function turn_red(slot){
    const canvas = document.getElementById('parkingCanvas');
    const ctx = canvas.getContext('2d');
    box = toado[slot]

    ctx.clearRect(box.x, box.y, box.w , box.h );

    ctx.strokeStyle = '#FF0000'
    ctx.lineWidth = 2;
    ctx.strokeRect(box.x, box.y, box.w, box.h);
    ctx.fillStyle ='rgba(204, 32, 32, 0.5)';
    ctx.fillRect(box.x, box.y, box.w, box.h);
}

document.getElementById("book_btn").addEventListener("click", function() {
    const slot = document.getElementById("parking_slot").value;
    turn_red(slot)
});

setInterval(read_toado, 1000)

main()