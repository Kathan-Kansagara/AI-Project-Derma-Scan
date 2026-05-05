const fs = require('fs');
const path = require('path');

const datasetPath = path.join(__dirname, 'dataset');

const folders = fs.readdirSync(datasetPath);

folders.forEach(folder => {
  const folderPath = path.join(datasetPath, folder);
  const images = fs.readdirSync(folderPath);

  console.log(`Disease: ${folder}`);
  console.log(`Images: ${images.length}`);
});