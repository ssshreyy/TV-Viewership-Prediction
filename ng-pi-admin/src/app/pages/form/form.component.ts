import { Component, OnInit } from '@angular/core';

@Component({
  selector: 'app-form',
  template: `<router-outlet></router-outlet>`
})
export class FormComponent implements OnInit {

  serverData: JSON;
  employeeData: JSON;
  employee:JSON;

  constructor() { }

  ngOnInit() {
  }
}
