'use strict';
// Exercise replay calculations without a browser, network, or DOM snapshots.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../laboratory/static/app.js'), 'utf8');
const bootstrap = source.indexOf("$('primary').addEventListener");
assert.ok(bootstrap > 0);
const context = vm.createContext({document: {getElementById: () => ({checked: true})}});
vm.runInContext(source.slice(0, bootstrap), context);
function evaluate(expression) { return JSON.parse(vm.runInContext(`JSON.stringify(${expression})`, context)); }
function prepare(script) { vm.runInContext(script, context); }

prepare(`
var fixture = {metadata: {role: 'weak'}, events: [
  {event_type: 'EpisodeStarted', payload: {snapshot: {text: 'HIDDEN WORLD'}}},
  {event_type: 'ObservationReceived', payload: {observation: {text: 'Visible tree'}, available_actions: [{name: 'DO'}]}},
  {event_type: 'ModelCalled', payload: {call_index: 0, model_role: 'weak', request: {model: 'candidate', observation: {text: 'Visible tree'}, available_actions: [{name: 'DO'}]}}},
  {event_type: 'ModelResponded', payload: {call_index: 0, latency_ms: 9, response: {model: 'candidate', usage: {input_tokens: 11, output_tokens: 2}}}},
  {event_type: 'ModelCalled', payload: {call_index: 1, model_role: 'weak', request: {model: 'candidate'}}},
  {event_type: 'ModelCallFailed', payload: {call_index: 1}},
  {event_type: 'ActionRejected', payload: {proposal: 'UNKNOWN_ACTION', reason: 'not advertised'}}
]};
`);
const partial = evaluate('accumulated(fixture, 3)');
assert.equal(partial.text, 'Visible tree');
assert.equal(partial.calls, 1);
assert.equal(partial.usage[0].total, 11);
assert.equal(partial.usage[2].total, null);
assert.equal(partial.usage[2].unknown, 1);
const failed = evaluate('accumulated(fixture, 6)');
assert.equal(failed.calls, 2);
assert.equal(failed.usage[0].total, null);
assert.equal(failed.usage[0].known, 11);
assert.equal(failed.usage[0].unknown, 1);
assert.equal(failed.accepted, false);
assert.equal(failed.feedback, 'not advertised');
assert.equal(evaluate('accumulated(fixture, 0)').text, '');

prepare(`state.primary = {events: [{step: 0}, {step: 1}, {step: 1}, {step: 2}, {step: 3}]};
state.comparison = {events: [{step: 0}, {step: 1}, {step: 1}, {step: 1}, {step: 2}]};
state.index = 2;`);
assert.equal(evaluate('comparisonIndex()'), 2);
prepare('state.index = 4;');
assert.equal(evaluate('comparisonIndex()'), 4);
console.log('Replay logic: visible-state isolation, unknown usage, failed-call accounting, rejection and step synchronization passed.');
