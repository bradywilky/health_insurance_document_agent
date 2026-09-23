// Synthetic workbook fixture. Requires @oai/artifact-tool in the Node environment.
import fs from 'node:fs/promises';
import path from 'node:path';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

const outputDir = process.argv[2] ?? 'outputs/fishing';
await fs.mkdir(outputDir, { recursive: true });
const wb = Workbook.create();
const specs = [];
function add(name, rows, headerRow = null, widths = []) {
  const sheet = wb.worksheets.add(name);
  const cols = Math.max(...rows.map(r => r.length));
  const values = rows.map(r => [...r, ...Array(cols-r.length).fill(null)]);
  sheet.getRangeByIndexes(0, 0, rows.length, cols).setNumberFormat('@');
  sheet.getRangeByIndexes(0, 0, rows.length, cols).values = values;
  sheet.showGridLines = false;
  const range = sheet.getRangeByIndexes(0, 0, rows.length, cols);
  range.format.font = { name: 'Arial', size: 10 };
  range.format.columnWidth = 24;
  range.format.rowHeight = 42;
  range.format.wrapText = true;
  range.format.verticalAlignment = 'center';
  widths.forEach((width, col) => sheet.getRangeByIndexes(0, col, rows.length, 1).format.columnWidth = width);
  if (headerRow) {
    sheet.getRangeByIndexes(headerRow-1, 0, 1, cols).format = {
      fill: '#254D32', font: { bold: true, color: '#FFFFFF', name: 'Arial', size: 10 }, wrapText: true,
    };
    sheet.freezePanes.freezeRows(headerRow);
  }
  specs.push({ name, rows: rows.length, cols });
  return sheet;
}
const about = add('About', [
  ['Fishing product field guide'],
  ['All products, observations, versions, and rules in this workbook are fictional test data.'],
  ['Purpose: align catch attributes across CastLog, RiverTrack, and TournamentDesk.'],
  ['Use Canonical ID to connect equivalent attributes. Similar labels alone do not prove equivalence.'],
  ['A blank target means no mapping is defined. Do not invent a mapping.'],
  ['Units matter: catch mass is standardized in grams; length is standardized in centimeters.'],
  ['Draft mappings are excluded from approved-only answers. Deprecated mappings are historical.'],
  ['Duplicate mappings are retained intentionally. Report them instead of silently counting twice.'],
  ['TournamentDesk does not capture water temperature in version 2.1.'],
  ['Privacy rule: exact fishing locations are rounded to a 1 km grid before export.'],
], null, [105]);
about.getRange('A1').format.font = { name:'Arial', size:14, bold:true };
const versions = add('Version History', [
  ['Version','Date','Change'],
  ['1.9','2026-01-10','Legacy length measurement used inches in RiverTrack.'],
  ['2.0','2026-02-10','RiverTrack switched fish_length to centimeters.'],
  ['2.1','2026-03-10','TournamentDesk added released_flag; temperature remains unsupported.'],
], 1, [15,18,85]);
versions.getRange('A2:A4').setNumberFormat('0.0');
add('Attribute Dictionary', [
  ['Canonical ID','Canonical Name','Definition','Unit'],
  ['catch_id','Catch identifier','Text identifier; retain leading zeros.','text'],
  ['species_code','Species code','Use Species Codes to interpret the code.','text'],
  ['mass_g','Catch mass','Mass of one fish in grams.','g'],
  ['length_cm','Fish length','Measured total length.','cm'],
  ['released','Released','Whether the fish was returned to the water.','boolean'],
  ['water_temp_c','Water temperature','Water temperature when caught.','C'],
], 1, [22,24,55,16]);
const headers = ['Canonical ID','Source Field','Source Unit','Target Field','Status','Notes'];
const cast = add('CastLog Mapping', [
  ['CastLog integration worksheet'],
  ['Version 2.1 - fishing export'],
  [], headers,
  ['catch_id','catch_no','text','catch_id','Approved','Keep leading zeros'],
  ['species_code','species','code','species_code','Approved','See Species Codes'],
  ['mass_g','weight_oz','oz','mass_g','Approved','Multiply by 28.3495'],
  [],
  ['length_cm','length_in','in','length_cm','Approved','Multiply by 2.54'],
  headers,
  ['released','release_status','Y/N','released','Approved','Y=true; N=false'],
  ['water_temp_c','water_f','F','water_temp_c','Draft','(F - 32) * 5 / 9'],
  ['Note: Draft temperature mapping is awaiting field validation.'],
], 4, [22,22,18,22,18,40]);
cast.mergeCells('A1:F1');
add('RiverTrack Mapping', [
  headers,
  ['catch_id','record_id','text','catch_id','Approved','Keep leading zeros'],
  ['species_code','fish_code','code','species_code','Approved','See Species Codes'],
  ['mass_g','fish_weight','g','mass_g','Approved','No conversion'],
  ['length_cm','fish_length','cm','length_cm','Approved','Changed in version 2.0'],
  ['released','returned','0/1','released','Approved','1=true; 0=false'],
  ['water_temp_c','temp_c','C','water_temp_c','Approved','No conversion'],
  ['mass_g','fish_weight','g','mass_g','Approved','No conversion'],
], 1, [22,22,18,22,18,40]);
add('TournamentDesk Mapping', [
  ['TournamentDesk export'], [],
  ['Canonical ID','Source Field','Source Unit','Target Field','Status','Notes','Notes',null],
  ['catch_id','entry_id','text','catch_id','Approved','Keep leading zeros','Public export',null],
  ['species_code','species_id','code','species_code','Approved','See Species Codes',null,null],
  ['mass_g','weight','lb','mass_g','Approved','Multiply by 453.592','Scale reading',null],
  ['mass_g','weight','oz','mass_g','Draft','Multiply by 28.3495','Conflicts with approved lb unit',null],
  ['length_cm','size_in','in','length_cm','Approved','Multiply by 2.54',null,null],
  ['released','released_flag','Y/N','released','Approved','Added in version 2.1',null,null],
  ['water_temp_c',null,null,null,'Unmapped','Not captured in version 2.1',null,null],
], 3, [22,22,18,22,18,40,40,12]);
add('Species Codes', [
  ['Code','Species','Habitat'],
  ['LMB','Largemouth bass','Freshwater'],
  ['SMB','Smallmouth bass','Freshwater'],
  ['RBT','Rainbow trout','Freshwater'],
  ['NA','Not assessed','Unknown'],
], 1, [18,35,22]);
const catches = add('Catch Samples', [
  ['Catch ID','Product','Species Code','Raw Mass','Unit','Released'],
  ['0012','CastLog','LMB',16,'oz','Y'],
  ['0013','RiverTrack','SMB',500,'g','1'],
  ['0014','TournamentDesk','LMB',2,'lb','N'],
  ['0015','CastLog','NA',null,'oz','Y'],
  ['0016','RiverTrack','RBT',0,'g','0'],
], 1, [18,25,20,18,18,18]);
catches.getRange('A2:A6').setNumberFormat('0000');
wb.recalculate();
console.log((await wb.inspect({kind:'sheet',include:'id,name'})).ndjson);
for (const spec of specs) {
  const preview = await wb.render({sheetName:spec.name,range:`A1:${String.fromCharCode(64+spec.cols)}${spec.rows}`,scale:1,format:'png'});
  await fs.writeFile(path.join(outputDir,`${spec.name}.png`),new Uint8Array(await preview.arrayBuffer()));
}
const file = await SpreadsheetFile.exportXlsx(wb);
await file.save(path.join(outputDir,'fishing_mappings.xlsx'));
console.log(`Saved ${path.join(outputDir,'fishing_mappings.xlsx')}`);
