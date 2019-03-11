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
  arrayOfKeys;
  pageSize = 10;
  pageNumber = 1;

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
    var body = "username=" + tweet.username + "&query=" + tweet.query + "&since=" + tweet.since + "&until=" + tweet.until + "&maxNo=" + tweet.maxNo +"&top=" + tweet.top;
    this.httpClient.post("http://127.0.0.1:5003/search", body, httpOptions).subscribe((data) => {
      this.serverData = data as JSON;
      // alert(JSON.stringify(this.serverData))
      console.log(this.serverData)
      // alert(typeof this.serverData)
      this.arrayOfKeys = Object.keys(this.serverData)
      // alert(this.arrayOfKeys)
    })
  }

  pageChanged(pN: number): void {
    this.pageNumber = pN;
  }

}
