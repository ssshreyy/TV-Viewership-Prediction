import { Component, OnInit } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { HttpHeaders } from '@angular/common/http';

@Component({
  selector: 'app-form-layouts',
  templateUrl: './form-layouts.component.html',
  styleUrls: ['./form-layouts.component.scss']
})

export class FormLayoutsComponent implements OnInit {

  serverData: JSON;
  employeeData: JSON;
  employee:JSON;

  // constructor(){}
  constructor(private httpClient: HttpClient) {
  }
  
  ngOnInit() {
  }

  // searchTweet() {
  //   this.httpClient.post('http://127.0.0.1:5003/search', {'message': 123})
  // }

  searchTweet(tweet) {
    console.log(tweet);
    // alert(tweet.username)
    const httpOptions = {
      headers: new HttpHeaders({
        'Content-Type':  'application/x-www-form-urlencoded',
        'Authorization': 'my-auth-token'
      })
    };
    var body = "username=" + tweet.username + "&query=" + tweet.query + "&since=" + tweet.since + "&until=" + tweet.until + "&maxNo=" + tweet.maxNo;
    this.httpClient.post("http://127.0.0.1:5003/search", body, httpOptions).subscribe((data) => {
      this.serverData = data as JSON;
      alert(JSON.stringify(data))
      console.log(data)
    })
  }

  sayHi() {
    this.httpClient.get('http://127.0.0.1:5003/').subscribe(data => {
      this.serverData = data as JSON;
      console.log(this.serverData);
    })
  }

  getAllEmployees() {
    this.httpClient.get('http://127.0.0.1:5003/employees').subscribe(data => {
      this.employeeData = data as JSON;
      console.log(this.employeeData);
    })
  }
  getEmployee() {
    this.httpClient.get('http://127.0.0.1:5003/employees/1').subscribe(data => {
      this.employee = data as JSON;
      console.log(this.employee);
    })
  }

}
