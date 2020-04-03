const setTilesSize = () => {
    window.requestAnimationFrame(() => {
      const width = document.querySelector('.tile').clientWidth;
      document.querySelectorAll('.tile').forEach((tile) => {
        if (tile.clientHeight !== width) {
          tile.style.height = width + 'px';
          tile.style.fontSize = width + 'px';
          tile.style.lineHeight = width + 'px';
        }
      });
    });
  };
  
  const onMouseOverTile = (event) => {
    const target = event.currentTarget;
    const x = target.getAttribute('data-x');
    const y = target.getAttribute('data-y');
    window.requestAnimationFrame(() => {
      document.querySelector('.board').querySelectorAll('.tile').forEach((tile) => {
        if (tile.getAttribute('data-x') === x || tile.getAttribute('data-y') === y) {
          tile.classList.add('active-line');
        } else {
          tile.classList.remove('active-line');
        }
      });
    });
  };
  
  const onMouseLeftBoard = () => {
    window.requestAnimationFrame(() => {
      document.querySelector('.board').querySelectorAll('.tile').forEach((tile) => {
        tile.classList.remove('active-line');
      });
    });
  };
  const solveSudoku = () => {
    var final_sudoku = [];
    var final_string = "";
    for (let y=0; y<9; y++){
        final_sudoku.push([]);
        for (let x=0; x<9; x++){
            const tile = document.getElementById(y + '--' + x);
            var value = tile.innerHTML;
            if (value==="")
                value = "0";
            final_string += value;
            final_sudoku[y].push(tile.innerHTML);
        }
    }
    console.log(final_sudoku);
    console.log(final_string);
    $.post("/solve", {
        matrix: final_string
    },
    function(data, status){
        matrix_temp = (data);
        console.log(matrix_temp);
        if (matrix_temp[0][0] === 0){
          console.log("okay");
          alert("Please check the sudoku as it is not solvable in the current state");
        }
        else {
          matrix = matrix_temp;
          updateGrid(); 
          alert("Sudoku has been solved");
        }
        
    }
    );
  };

  const updateGrid = () => {
    
    for (let y = 0; y < 9; y++) {
        for (let x = 0; x < 9; x++) {
            const tile = document.getElementById(y + '--' + x);
            tile.innerHTML = matrix[y][x];
            tile.setAttribute('contentEditable', "false");
        }
    }
  }
  
  const loadGrid = () => {
    const root = document.getElementById('grid-root');
    const board = document.createElement('div');
    const submitButton = document.getElementById('solveButton');
    submitButton.addEventListener('click', solveSudoku);
    board.classList.add('board');
    for (let y = 0; y < 9; y++) {
      for (let x = 0; x < 9; x++) {
        const tile = document.createElement('div');
        tile.classList.add('tile');
        tile.id = y + '-' + x;
        tile.setAttribute('data-x', x);
        tile.setAttribute('data-y', y);
        tile.setAttribute('contentEditable', "true");
        tile.addEventListener('mouseover', onMouseOverTile);
        const content = document.createElement('div');
        content.id = y + '--' + x;
        content.classList.add('content');
        if (matrix[y][x] !== -1)
            content.innerHTML = matrix[y][x];
        tile.appendChild(content);
        board.appendChild(tile);
      }
    }
    board.addEventListener('mouseout', onMouseLeftBoard);
    root.appendChild(board);
    setTilesSize();
    window.addEventListener('resize', setTilesSize);
  };
  
  if (document.readyState !== 'loading') {
    loadGrid();
  } else {
    document.addEventListener('DOMContentLoaded', loadGrid);
  };