import { MailerService } from '@nestjs-modules/mailer';
import { Injectable } from '@nestjs/common';

@Injectable()
export class EmailService {
    constructor(private readonly mailService: MailerService) {}

    async sendTest(email:string){
        await this.mailService.sendMail({
            to:email,
            subject:"hello its a test",
            text:"this works",
            html: `<p>Hello ${email}</p>`,
            from: '"Generator Frajdy"',
        })
    }
    async sendVerificationMail(email:string, verifyUrl:string){
        await this.mailService.sendMail({
            to: email,
            subject: 'Verify your email',
            html: `<p>Click <a href="${verifyUrl}">here</a> to verify your account.</p>`,
            from: '"Generator Frajdy"',
        })
    }
}
